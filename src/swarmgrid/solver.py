"""Space-time A* planning for a single agent plus joint resolution for a mission.

The public contract of this module is intentionally small:

* :func:`distance_map` -- static BFS distances to a goal, ignoring other agents.
* :func:`space_time_astar` -- one-agent search in a time-expanded grid.
* :func:`detect_conflicts` -- vertex and edge-swap conflicts in a finished plan.
* :func:`plan_mission` -- joint planning for a mission.

Joint planning has two stages, in this order:

1. **Prioritized planning.** Agents are planned in priority order; each low-level
   search treats the paths of all higher-priority agents as moving obstacles.
   When an order fails, the next order in a deterministic permutation sequence is
   tried. This is cheap and resolves the common cases.
2. **Constraint tree.** If no priority order succeeds, a conflict-tree search
   (the outer loop of conflict-based search) branches on a concrete conflict:
   one child forbids one agent the conflicted cell or edge at that timestep. The
   tree is best-first by total cost, so it never misses a plan that an
   intermediate priority order hid behind a bad first choice. It is bounded by
   `tree_budget` nodes.

A mission is reported `unsolvable` only when the priority orders are exhausted
**and** the constraint tree is either fully explored without a solution or hits
its node budget. No path data is ever fabricated.

Determinism: every ordering, tie-break, and list in this module derives from
sorted agent identifiers, so the same map and mission always yield one plan.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

from .model import Agent, Closure, Mission, Point

Move = tuple[int, int]
Path = tuple[Point, ...]
Closures = tuple[Closure, ...]
ConstraintSet = dict[str, tuple[frozenset[tuple[Point, int]], frozenset[tuple[Point, int, Point]]]]

DEFAULT_ORDER_BUDGET = 24
DEFAULT_TREE_BUDGET = 64
DEFAULT_HOLD_BUDGET = 12
MAX_EXPANSIONS = 200_000
# A schedule never needs more ticks than this: every low-level search runs with
# this as its horizon, and the planner only ever asks for paths that start from
# the mission's own start cells, so the bound also caps how long an impossible
# mission is searched before the verdict is unsolvable.
MAX_TIMESTEP = 40

# Expansion order is fixed and deterministic -- wait first, then the four moves;
# ties in the A* heap fall back to the insertion counter, so no set/dict
# iteration order can leak into a plan.
DIRECTIONS: tuple[Move, ...] = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1))


def _closure_cell_set(closures: Closures, timestep: int) -> frozenset[Point]:
    """Cells every agent is forbidden to occupy at `timestep` by a closure."""
    blocked: set[Point] = set()
    for closure in closures:
        if closure.start_tick <= timestep <= closure.end_tick:
            blocked.update(closure.cells)
    return frozenset(blocked)


def _closure_start_conflict(
    agent: Agent, closures: Closures, tick_offset: int = 0
) -> Conflict | None:
    """A closure that already covers the agent's start at t=0 blocks the plan.

    The first state of every time-indexed path is the start cell at tick 0, so
    such an agent has no legal first state at all; name the closure directly.
    `tick_offset` only relabels the diagnostic: a replan searches in relative
    time but reports the absolute tick the snapshot was taken at.
    """
    covering = [
        closure
        for closure in closures
        if closure.start_tick == 0 and agent.start in closure.cells_set
    ]
    if not covering:
        return None
    return Conflict(
        "closure",
        tick_offset,
        agent.id,
        None,
        agent.start,
        None,
        f"agent {agent.id} is blocked on its start {list(agent.start)} at t={tick_offset} "
        "by a temporary closure",
    )


def _closure_block_conflict(
    agent: Agent,
    closures: Closures,
    closure_prune: tuple[int, Point] | None = None,
    tick_offset: int = 0,
) -> Conflict:
    """Name the closure that kept `agent` off its goal.

    `closure_prune`, when the search actually saw one, is the earliest tick (then
    smallest cell) a closure pruned; report that concrete block. Otherwise fall
    back to the closure whose interval starts earliest. `tick_offset` relabels
    the reported ticks from search-relative to absolute time.
    """
    if closure_prune is not None:
        timestep, cell = closure_prune
        interval = next(
            closure
            for closure in closures
            if closure.start_tick <= timestep <= closure.end_tick and cell in closure.cells_set
        )
        end_tick = interval.end_tick
    else:
        interval = min(closures, key=lambda item: (item.start_tick, item.end_tick))
        timestep = interval.start_tick
        end_tick = interval.end_tick
        cell = min(interval.cells)
    return Conflict(
        "closure",
        timestep + tick_offset,
        agent.id,
        None,
        cell,
        None,
        f"agent {agent.id} cannot reach goal {list(agent.goal)}: a temporary closure blocks "
        f"{list(cell)} from t={timestep + tick_offset} to t={end_tick + tick_offset}",
    )


@dataclass(frozen=True)
class Conflict:
    """A collision between two agents, or an impossible single-agent goal."""

    type: str
    timestep: int
    agent: str
    other: str | None
    position: Point
    other_position: Point | None
    message: str

    def as_dict(self) -> dict:
        return {
            "type": self.type,
            "timestep": self.timestep,
            "agent": self.agent,
            "other": self.other,
            "position": list(self.position),
            "other_position": list(self.other_position) if self.other_position is not None else None,
            "message": self.message,
        }

    def as_key(self) -> tuple:
        return (
            self.type,
            self.timestep,
            self.agent,
            self.other or "",
            self.position,
            self.other_position,
        )


# --------------------------------------------------------------------------- #
# static geometry
# --------------------------------------------------------------------------- #


def distance_map(map_record, goal: Point) -> list[int | None]:
    """Breadth-first move distances from every cell to `goal` on the static grid.

    Obstacle cells and cells that cannot reach `goal` are `None`. Other agents
    are ignored, which makes this a valid (admissible) lower bound.
    """
    distances: list[int | None] = [None] * (map_record.width * map_record.height)
    if not map_record.walkable(goal):
        return distances
    distances[map_record.index(goal)] = 0
    frontier = [goal]
    while frontier:
        following: list[Point] = []
        for cell in frontier:
            next_distance = distances[map_record.index(cell)] + 1  # type: ignore[operator]
            for neighbor in map_record.neighbors(cell):
                index = map_record.index(neighbor)
                if distances[index] is None:
                    distances[index] = next_distance
                    following.append(neighbor)
        frontier = following
    return distances


def _heuristic(cell: Point, goal: Point) -> int:
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


# --------------------------------------------------------------------------- #
# single-agent space-time A*
# --------------------------------------------------------------------------- #


def position_at(path: Path, timestep: int) -> Point | None:
    """Cell occupied at `timestep`, or `None` once the agent has finished.

    An agent's plan ends on its goal: it is done after `len(path) - 1` ticks, so
    it occupies no cell for every later timestep. A finished agent therefore
    never conflicts with an agent that still runs, and two agents can never
    settle on one cell.
    """
    if 0 <= timestep < len(path):
        return path[timestep]
    return None


def space_time_astar(
    map_record,
    agent: Agent,
    planned: list[tuple[str, Path]] | None = None,
    distance_to_goal: list[int | None] | None = None,
    forbidden_cells: frozenset[tuple[Point, int]] = frozenset(),
    forbidden_edges: frozenset[tuple[Point, int, Point]] = frozenset(),
    hold: int = 0,
    closures: Closures = (),
    tick_offset: int = 0,
) -> tuple[Path | None, Conflict | None]:
    """Plan `agent` in space and time.

    The returned path is time indexed: `path[t]` is the cell the agent occupies
    at timestep `t`, so it stays contiguous and ends on the goal. A plan may
    include waits, and once the agent has finished it reads as `None` from
    :func:`position_at`, so a finished agent blocks nothing.

    `hold` makes the agent stay on its start cell for that many ticks before it
    may move anywhere else. The joint planner uses it to explore "let the other
    agent go first" without forbidding the start cell outright.

    Pruned during expansion:

    * `planned` -- paths of higher-priority agents;
    * `forbidden_cells` -- `(cell, timestep)` the agent may not occupy;
    * `forbidden_edges` -- `(from_cell, timestep, to_cell)` the agent may not
      traverse; both directions of a traded edge are forbidden together;
    * `closures` -- task-level temporary closures: on the closed tick interval
      the listed cells may not be moved into, waited on, or held at all.

    The search itself always runs in relative time starting at tick 0;
    `tick_offset` is added to the timesteps of any reported conflict so a
    replan from a later snapshot names absolute ticks.

    Returns `(path, None)` on success, or `(None, conflict)` naming the blocking
    constraint when the goal cannot be reached under them.
    """
    planned = planned or []
    start_blocked = _closure_start_conflict(agent, closures, tick_offset)
    if start_blocked is not None:
        return None, start_blocked
    if distance_to_goal is None:
        distance_to_goal = distance_map(map_record, agent.goal)
    index_of_start = map_record.index(agent.start)
    if distance_to_goal[index_of_start] is None:
        return None, Conflict(
            "unreachable_goal",
            tick_offset,
            agent.id,
            None,
            agent.start,
            None,
            f"agent {agent.id} cannot reach its goal from {list(agent.start)} on the static grid",
        )

    def occupant(cell: Point, timestep: int, came_from: Point) -> tuple[str, Point] | None:
        """A higher-priority agent that makes this move a vertex conflict or a swap.

        Occupancy is read with :func:`position_at`, so a higher-priority agent
        blocks exactly the cells it holds on exactly the ticks it holds them --
        it stops blocking once it finishes on its goal. Following is legal:
        entering a cell that another agent leaves on the same tick never puts two
        agents on one cell.
        """
        for other_id, path in planned:
            if position_at(path, timestep) == cell:
                return (other_id, cell)
            if position_at(path, timestep - 1) == cell and position_at(path, timestep) == came_from:
                # The other agent moves from `cell` into `came_from` on this very
                # tick, so this move would trade the two agents' cells.
                return (other_id, cell)
        return None

    start = agent.start
    counter = 0
    # States are (cell, timestep) pairs: with time-indexed constraints a cell may
    # be usable at one timestep and blocked at another, so arrival time is part of
    # the state. The heap is ordered by estimated total cost, and `came_from` is
    # keyed by state so a wait and a move into the same cell stay distinct.
    heap: list[tuple[int, int, int, Point]] = [(distance_to_goal[index_of_start], 0, counter, start)]
    came_from: dict[tuple[Point, int], tuple[Point, int] | None] = {(start, 0): None}
    reached: set[tuple[Point, int]] = {(start, 0)}
    expansions = 0
    # The earliest tick (then smallest cell) at which a closure pruned a move the
    # search actually considered; that is the concrete obstruction to report if
    # no route survives.
    closure_prune: tuple[int, Point] | None = None

    while heap:
        _, timestep, _, cell = heapq.heappop(heap)
        if timestep > MAX_TIMESTEP:
            continue
        expansions += 1
        if expansions > MAX_EXPANSIONS:
            break
        if (
            cell == agent.goal
            and (cell, timestep) not in forbidden_cells
            and cell not in _closure_cell_set(closures, timestep)
        ):
            return _fill_waits(came_from, (cell, timestep), agent.goal), None

        following = timestep + 1
        closed_cells = _closure_cell_set(closures, following)
        for move in DIRECTIONS:
            neighbor = (cell[0] + move[0], cell[1] + move[1])
            if not map_record.walkable(neighbor):
                continue
            if timestep < hold and neighbor != start:
                continue
            if (neighbor, following) in forbidden_cells:
                continue
            if neighbor in closed_cells:
                # A temporary closure forbids entering, waiting on, or holding
                # this cell on this tick.
                candidate = (following, neighbor)
                if closure_prune is None or candidate < closure_prune:
                    closure_prune = candidate
                continue
            if (cell, following, neighbor) in forbidden_edges:
                continue
            if (neighbor, following) in reached:
                continue
            if occupant(neighbor, following, cell) is not None:
                continue
            reached.add((neighbor, following))
            came_from[(neighbor, following)] = (cell, timestep)
            counter += 1
            heapq.heappush(
                heap,
                (distance_to_goal[map_record.index(neighbor)] + following, following, counter, neighbor),
            )

    return None, _exhausted(
        agent, planned, forbidden_cells, forbidden_edges, closures, closure_prune, tick_offset
    )


def _exhausted(
    agent: Agent,
    planned: list[tuple[str, Path]],
    forbidden_cells: frozenset[tuple[Point, int]],
    forbidden_edges: frozenset[tuple[Point, int, Point]],
    closures: Closures = (),
    closure_prune: tuple[int, Point] | None = None,
    tick_offset: int = 0,
) -> Conflict:
    """Explain an exhausted search; the usual causes are checked in order."""
    for other_id, path in planned:
        if path[-1] == agent.goal:
            return Conflict(
                "vertex",
                tick_offset,
                agent.id,
                other_id,
                agent.goal,
                agent.goal,
                f"agent {agent.id} cannot reach goal {list(agent.goal)}: it is also the goal of "
                f"{other_id}, whose plan occupies the route",
            )
    if forbidden_cells:
        timestep, cell = min((step, position) for position, step in forbidden_cells)
        return Conflict(
            "vertex",
            timestep + tick_offset,
            agent.id,
            None,
            cell,
            None,
            f"agent {agent.id} cannot reach goal {list(agent.goal)}: {list(cell)} is forbidden "
            f"at t={timestep + tick_offset}",
        )
    if forbidden_edges:
        timestep, edge = min((step, (source, target)) for source, step, target in forbidden_edges)
        return Conflict(
            "edge",
            timestep + tick_offset,
            agent.id,
            None,
            edge[0],
            edge[1],
            f"agent {agent.id} cannot reach goal {list(agent.goal)}: the edge "
            f"{list(edge[0])}->{list(edge[1])} is forbidden at t={timestep + tick_offset}",
        )
    if closures:
        return _closure_block_conflict(agent, closures, closure_prune, tick_offset)
    return Conflict(
        "unreachable_goal",
        tick_offset,
        agent.id,
        None,
        agent.goal,
        None,
        f"agent {agent.id} has no conflict-free route to goal {list(agent.goal)} under these constraints",
    )


def _fill_waits(
    came_from: dict[tuple[Point, int], tuple[Point, int] | None],
    current: tuple[Point, int],
    goal: Point,
) -> Path:
    """Rebuild the time-indexed path from the goal state back to the start."""
    cells: list[Point] = []
    node: tuple[Point, int] | None = current
    while node is not None:
        cells.append(node[0])
        node = came_from[node]
    cells.reverse()
    assert cells[-1] == goal
    return tuple(cells)


# --------------------------------------------------------------------------- #
# joint plan validation
# --------------------------------------------------------------------------- #


def detect_conflicts(
    order: tuple[Agent, ...], paths: list[Path], tick_offset: int = 0
) -> list[Conflict]:
    """Every vertex conflict and edge swap in a joint plan, ordered by timestep.

    `tick_offset` relabels the reported timesteps from path-relative to
    absolute time; the paths themselves are always indexed from 0.
    """
    found: list[Conflict] = []
    horizon = max((len(path) for path in paths), default=0)
    for timestep in range(horizon):
        for left in range(len(order)):
            first = position_at(paths[left], timestep)
            if first is None:
                continue
            for right in range(left + 1, len(order)):
                second = position_at(paths[right], timestep)
                if second is None:
                    continue
                if first == second:
                    found.append(
                        Conflict(
                            "vertex",
                            timestep + tick_offset,
                            order[left].id,
                            order[right].id,
                            first,
                            second,
                            f"{order[left].id} and {order[right].id} both occupy {list(first)} "
                            f"at t={timestep + tick_offset}",
                        )
                    )
                    continue
                if timestep + 1 >= horizon:
                    continue
                left_next = position_at(paths[left], timestep + 1)
                right_next = position_at(paths[right], timestep + 1)
                if left_next is None or right_next is None:
                    continue
                if left_next == second and right_next == first:
                    found.append(
                        Conflict(
                            "edge",
                            timestep + tick_offset,
                            order[left].id,
                            order[right].id,
                            first,
                            second,
                            f"{order[left].id} and {order[right].id} swap {list(first)} and "
                            f"{list(second)} at t={timestep + tick_offset}",
                        )
                    )
    return sorted(found, key=lambda conflict: conflict.as_key())


def makespan(paths: Iterable[Path]) -> int:
    return max((len(path) - 1 for path in paths), default=0)


# --------------------------------------------------------------------------- #
# joint planning
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlanResult:
    """`method` is `priority` or `constraint_tree`; `order` drives the agent listing."""

    status: str
    method: str
    order: tuple[str, ...]
    paths: tuple[Path, ...]
    conflicts: tuple[Conflict, ...]
    orders_tried: int
    orders_budget: int

    @property
    def solution_makespan(self) -> int:
        return makespan(self.paths)


def plan_mission(
    map_record,
    mission: Mission,
    requested_priority: list[str] | None = None,
    order_budget: int = DEFAULT_ORDER_BUDGET,
    tree_budget: int = DEFAULT_TREE_BUDGET,
    closures: Closures = (),
    tick_offset: int = 0,
) -> PlanResult:
    """Resolve `mission` with prioritized planning, then a constraint tree.

    `map_record` is duck-typed: it needs `width`, `height`, `walkable`, `index`,
    and `neighbors` (that is, :class:`swarmgrid.model.Map`).

    `closures` are task-level temporary closures shared by every agent: on each
    closure's closed tick interval its cells act as space-time obstacles for the
    whole fleet.

    The search runs in relative time: every path is indexed from tick 0 and the
    closures are expected in the same frame. `tick_offset` is added to the
    timesteps of every reported conflict so a replan that shifted its closures
    into relative time still names absolute ticks in its diagnostics.
    """
    agents = tuple(sorted(mission.agents, key=lambda agent: agent.id))
    orders_budget = _orders_budget(len(agents), order_budget, requested_priority)

    # A closure already covering a start at tick 0 denies the agent its first
    # state in every order and every tree node, so diagnose it directly once.
    start_blocked = [
        conflict
        for agent in agents
        if (conflict := _closure_start_conflict(agent, closures, tick_offset)) is not None
    ]
    if start_blocked:
        return PlanResult(
            status="unsolvable",
            method="none",
            order=(),
            paths=(),
            conflicts=tuple(start_blocked),
            orders_tried=0,
            orders_budget=orders_budget,
        )

    distance_from_goal: dict[Point, list[int | None]] = {
        agent.goal: distance_map(map_record, agent.goal) for agent in agents
    }
    unreachable = [
        Conflict(
            "unreachable_goal",
            tick_offset,
            agent.id,
            None,
            agent.goal,
            None,
            f"agent {agent.id} goal {list(agent.goal)} is unreachable on the static map",
        )
        for agent in agents
        if distance_from_goal[agent.goal][map_record.index(agent.start)] is None
    ]
    if unreachable:
        return PlanResult(
            status="unsolvable",
            method="none",
            order=(),
            paths=(),
            conflicts=tuple(sorted(unreachable, key=lambda conflict: conflict.as_key())),
            orders_tried=0,
            orders_budget=orders_budget,
        )

    budget = orders_budget
    if requested_priority is not None:
        sequence: list[tuple[str, ...]] = [tuple(requested_priority)]
    else:
        sequence = _order_sequence(tuple(agent.id for agent in agents), budget)

    conflicts: list[Conflict] = []
    tried = 0
    for order_ids in sequence:
        tried += 1
        order = tuple(next(agent for agent in agents if agent.id == agent_id) for agent_id in order_ids)
        paths, failure = _run_order(
            map_record, order, distance_from_goal, closures=closures, tick_offset=tick_offset
        )
        if paths is None:
            if failure is not None:
                conflicts.append(failure)
            continue
        joint = detect_conflicts(order, paths, tick_offset)
        if joint:
            conflicts.extend(joint)
            continue
        return PlanResult(
            status="solved",
            method="priority",
            order=order_ids,
            paths=tuple(paths),
            conflicts=(),
            orders_tried=tried,
            orders_budget=budget,
        )

    if requested_priority is None:
        solution = _constraint_tree(
            map_record, agents, distance_from_goal, tree_budget, budget, closures, tick_offset
        )
        if solution is not None:
            paths, tree_conflicts = solution
            if not tree_conflicts:
                return PlanResult(
                    status="solved",
                    method="constraint_tree",
                    order=tuple(agent.id for agent in agents),
                    paths=tuple(paths),
                    conflicts=(),
                    orders_tried=tried,
                    orders_budget=budget,
                )
            conflicts.extend(tree_conflicts)

    return PlanResult(
        status="unsolvable",
        method="none",
        order=(),
        paths=(),
        conflicts=tuple(_deduplicate(conflicts)),
        orders_tried=tried,
        orders_budget=budget,
    )


def _run_order(
    map_record,
    order: tuple[Agent, ...],
    distance_from_goal: dict[Point, list[int | None]],
    constraints: ConstraintSet | None = None,
    first_hold: int = 0,
    closures: Closures = (),
    tick_offset: int = 0,
) -> tuple[list[Path] | None, Conflict | None]:
    """Plan `order` one agent at a time, each treating the earlier paths as moving obstacles.

    Waiting is a legal move, but a step planner enters the grid as early as it
    can, so on a bottleneck it can walk into a pocket. Every agent is therefore
    offered a head start: it may hold its start cell for a few ticks before it
    enters the grid, and `first_hold` does the same for the highest-priority
    agent, whose early entrance can be what blocks everybody else. The first
    option that actually routes the agent is used.
    """
    constraints = constraints or {}
    planned: list[tuple[str, Path]] = []
    for index, agent in enumerate(order):
        cells, edges = constraints.get(agent.id, (frozenset(), frozenset()))
        if index == 0 and first_hold > 0:
            path, failure = space_time_astar(
                map_record,
                agent,
                planned,
                distance_from_goal[agent.goal],
                cells,
                edges,
                first_hold,
                closures,
                tick_offset,
            )
            if path is None:
                assert failure is not None
                return None, failure
        else:
            path, failure = _route_with_wait(
                map_record, agent, planned, distance_from_goal, cells, edges, closures, tick_offset
            )
            if path is None:
                assert failure is not None
                return None, failure
        planned.append((agent.id, path))
    return [path for _, path in planned], None


def _route_with_wait(
    map_record,
    agent: Agent,
    planned: list[tuple[str, Path]],
    distance_from_goal: dict[Point, list[int | None]],
    cells: frozenset[tuple[Point, int]],
    edges: frozenset[tuple[Point, int, Point]],
    closures: Closures = (),
    tick_offset: int = 0,
) -> tuple[Path | None, Conflict | None]:
    """Try routing `agent` after holding its start cell for 0..N ticks.

    When every option fails, a two-agent goal contention is reported as a vertex
    conflict on the shared goal: that is the concrete cause, and it is far more
    useful to a caller than the bare "no route" dead end.
    """
    failure: Conflict | None = None
    for hold in range(DEFAULT_HOLD_BUDGET + 1):
        path, conflict = space_time_astar(
            map_record, agent, planned, distance_from_goal[agent.goal], cells, edges, hold,
            closures, tick_offset
        )
        if path is not None:
            return path, None
        if failure is None:
            failure = conflict
    assert failure is not None
    # A closure denies a tick interval to everyone; when it is what exhausted the
    # search, the inter-agent heuristics below would mislabel the blockage.
    if failure.type == "closure":
        return None, failure
    for other_id, path in planned:
        if path[-1] == agent.goal:
            return None, Conflict(
                "vertex",
                tick_offset,
                agent.id,
                other_id,
                agent.goal,
                agent.goal,
                f"agent {agent.id} cannot reach goal {list(agent.goal)}: {other_id} is "
                f"planned to end there",
            )
    for other_id, path in planned:
        if path[-1] == agent.start:
            return None, Conflict(
                "vertex",
                len(path) - 1 + tick_offset,
                agent.id,
                other_id,
                agent.start,
                agent.goal,
                f"agent {agent.id} is trapped on {list(agent.start)}: {other_id} finishes "
                f"there at t={len(path) - 1 + tick_offset}",
            )
    return None, failure


def _order_sequence(default_order: tuple[str, ...], budget: int) -> list[tuple[str, ...]]:
    """The deterministic priority orders to try: the default first, then permutations."""
    budget = max(1, budget)
    ordered: list[tuple[str, ...]] = [default_order]
    for permutation in sorted(permutations(default_order)):
        if len(ordered) >= budget:
            break
        if permutation != default_order:
            ordered.append(permutation)
    return ordered


# --------------------------------------------------------------------------- #
# stage two: conflict tree (outer loop of conflict-based search)
# --------------------------------------------------------------------------- #


def _constraint_tree(
    map_record,
    agents: tuple[Agent, ...],
    distance_from_goal: dict[Point, list[int | None]],
    tree_budget: int,
    order_budget: int = DEFAULT_ORDER_BUDGET,
    closures: Closures = (),
    tick_offset: int = 0,
) -> tuple[list[Path], list[Conflict]] | None:
    """Best-first search over sets of per-agent space-time constraints.

    Returns `(paths, [])` for a conflict-free joint plan. When no plan is found it
    returns `(infeasible_paths, [conflict])` explaining why: the failure blocking
    the best node when the tree ran out of budget, or an empty conflict list when
    every node it reached was infeasible and nothing was left to branch on.

    Temporary `closures` constrain every node alike -- they are fleet-wide
    facts, not branching choices -- so they are passed straight to each node's
    low-level planner.
    """
    root: ConstraintSet = {}
    paths, failure = _plan_all(
        map_record, agents, distance_from_goal, root, order_budget, closures, tick_offset
    )
    if failure is None:
        return paths, []
    counter = 0
    heap: list[tuple[int, int, tuple]] = [
        (
            _paths_cost(paths) if len(paths) == len(agents) else INFEASIBLE_COST,
            counter,
            (root, paths, failure),
        )
    ]
    expanded = 0

    while heap:
        _, _, (constraints, paths, failure) = heapq.heappop(heap)
        if failure is None:
            return paths, []
        if expanded >= tree_budget:
            return paths, [failure]
        expanded += 1
        for child in _branch(constraints, failure, tick_offset):
            child_paths, child_failure = _plan_all(
                map_record, agents, distance_from_goal, child, order_budget, closures, tick_offset
            )
            counter += 1
            heapq.heappush(
                heap,
                (
                    _paths_cost(child_paths) if child_failure is None else INFEASIBLE_COST,
                    counter,
                    (child, child_paths, child_failure),
                ),
            )
    return paths, [failure]


INFEASIBLE_COST = 1 << 40


def _plan_all(
    map_record,
    agents: tuple[Agent, ...],
    distance_from_goal: dict[Point, list[int | None]],
    constraints: ConstraintSet,
    order_budget: int = DEFAULT_ORDER_BUDGET,
    closures: Closures = (),
    tick_offset: int = 0,
) -> tuple[list[Path], Conflict | None]:
    """Plan all agents at one node of the constraint tree.

    Because a node's leaf planner plans one agent at a time, which agent yields is
    a real choice: on a bottleneck one order finds a plan and the reverse order
    traps the lower-priority agent. Every node therefore tries the deterministic
    priority orders and keeps a conflict-free plan if one appears. When none does,
    the node reports its blocking conflict so the tree can branch on it.
    """
    fallback: Conflict | None = None
    report: list[Path] | None = None
    for order_ids in _order_sequence(tuple(agent.id for agent in agents), order_budget):
        order = tuple(next(agent for agent in agents if agent.id == agent_id) for agent_id in order_ids)
        for first_hold in range(DEFAULT_HOLD_BUDGET + 1):
            outcome, failure = _run_order(
                map_record, order, distance_from_goal, constraints, first_hold, closures,
                tick_offset
            )
            if outcome is not None:
                joint = detect_conflicts(order, outcome, tick_offset)
                if not joint:
                    ordered = {agent.id: path for agent, path in zip(order, outcome)}
                    return [ordered[agent.id] for agent in agents], None
                if report is None:
                    ordered = {agent.id: path for agent, path in zip(order, outcome)}
                    report = [ordered[agent.id] for agent in agents]
                conflict = joint[0]
            else:
                assert failure is not None
                conflict = failure
            if fallback is None:
                fallback = conflict
    # A plan that is complete but still conflicts names a concrete vertex or
    # swap, which is far more useful than the dead end of a single agent.
    if report is not None:
        conflicts = detect_conflicts(agents, report, tick_offset)
        if conflicts:
            return report, conflicts[0]
    return report or [], fallback


def _paths_cost(paths: list[Path]) -> int:
    return sum(len(path) - 1 for path in paths)


def _branch(
    constraints: ConstraintSet, conflict: Conflict, tick_offset: int = 0
) -> list[ConstraintSet]:
    """Children of a node: forbid the conflicted cell or edge to one of the agents.

    A two-agent conflict branches on both agents, so the tree covers "the first
    agent yields" and "the second one does". A conflict that involves a single
    agent is branched on that agent, and a bare `unreachable_goal` cannot be
    branched on at all.

    Conflict timesteps are labeled in absolute time (see `tick_offset`), while
    constraints are relative to the search frame, so the offset is removed here.
    """
    timestep = conflict.timestep - tick_offset
    if conflict.other is None:
        if conflict.type in ("unreachable_goal", "closure"):
            # A statically unreachable goal and a temporary closure are facts
            # about the fleet, not disagreements between agents: no child
            # constraint can make either one go away, so the node is a leaf.
            return []
        pairs = [(conflict.agent, True)]
    else:
        pairs = [
            (conflict.agent if conflict.agent < conflict.other else conflict.other, False),
            (conflict.other if conflict.agent < conflict.other else conflict.agent, False),
        ]
    children: list[ConstraintSet] = []
    for agent_id, single_agent in pairs:
        child = {key: (set(cells), set(edges)) for key, (cells, edges) in constraints.items()}
        cells, edges = child.setdefault(agent_id, (set(), set()))
        if single_agent:
            if timestep == 0:
                continue
            edges.add((conflict.position, timestep - 1, conflict.position))
        elif conflict.type == "vertex":
            cells.add((conflict.position, timestep))
        else:
            other_position = conflict.other_position or conflict.position
            edges.add((conflict.position, timestep, other_position))
            edges.add((other_position, timestep, conflict.position))
        children.append(
            {key: (frozenset(value[0]), frozenset(value[1])) for key, value in sorted(child.items())}
        )
    return children


def _deduplicate(conflicts: list[Conflict]) -> list[Conflict]:
    seen: dict[tuple, Conflict] = {}
    for conflict in conflicts:
        seen.setdefault(conflict.as_key(), conflict)
    return sorted(seen.values(), key=lambda conflict: conflict.as_key())


def _orders_budget(agent_count: int, budget: int, requested: list[str] | None) -> int:
    if requested is not None:
        return 1
    if agent_count <= 1:
        return 1
    return max(1, min(budget, _factorial(agent_count)))


def _factorial(value: int) -> int:
    total = 1
    for factor in range(2, value + 1):
        total *= factor
    return total
