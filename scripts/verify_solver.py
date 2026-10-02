"""Throwaway verification harness for the swarmgrid seed (not part of the repo).

Phase 1 checks every documented invariant on hundreds of randomly generated maps.
Phase 2 compares `unsolvable` verdicts against an exhaustive joint-path search on
small maps, to catch a planner that gives up while a valid plan exists.
"""

import itertools
import random

from swarmgrid.model import Map, Mission
from swarmgrid.solver import detect_conflicts, plan_mission, position_at

def assert_solved(map_rec, mission, result):
    order = mission.order(list(result.order))
    paths = list(result.paths)
    assert len(paths) == len(order)
    for agent, path in zip(order, paths):
        assert path[0] == agent.start, (agent.id, path)
        assert path[-1] == agent.goal, (agent.id, path)
        for index, cell in enumerate(path):
            assert map_rec.walkable(cell), (agent.id, cell)
            if index:
                previous = path[index - 1]
                assert abs(previous[0] - cell[0]) + abs(previous[1] - cell[1]) <= 1
    conflicts = detect_conflicts(order, paths)
    assert not conflicts, conflicts
    assert result.solution_makespan == max(len(path) - 1 for path in paths)
    goals = {path[-1] for path in paths}
    assert len(goals) == len(paths), "two agents settle on one cell"


def physically_sound(paths):
    """Replay a joint plan with the same rules the service documents.

    Two agents may not share a cell; they may not swap cells along an edge; and a
    waiting agent may not be crossed through by an agent moving into its cell.
    """
    horizon = max(len(path) for path in paths)
    for timestep in range(horizon + 1):
        here = [position_at(path, timestep) for path in paths]
        live = [cell for cell in here if cell is not None]
        if len(set(live)) != len(live):
            return False
        nxt = [position_at(path, timestep + 1) for path in paths]
        for left in range(len(paths)):
            for right in range(left + 1, len(paths)):
                if here[left] is None or here[right] is None:
                    continue
                if nxt[left] is None or nxt[right] is None:
                    continue
                if nxt[left] == here[right] and nxt[right] == here[left]:
                    return False
                if here[left] == nxt[left] and nxt[right] == here[left]:
                    return False
                if here[right] == nxt[right] and nxt[left] == here[right]:
                    return False
    return True


def brute(map_rec, mission, horizon=8):
    """Exhaustive joint search; returns a conflict-free plan or None."""
    agents = sorted(mission.agents, key=lambda agent: agent.id)
    all_paths = []
    for agent in agents:
        single = []
        stack = [(agent.start, (agent.start,))]
        while stack:
            cell, path = stack.pop()
            if len(path) - 1 > horizon:
                continue
            if cell == agent.goal:
                single.append(path)
                continue
            for neighbor in map_rec.neighbors(cell):
                stack.append((neighbor, path + (neighbor,)))
            stack.append((cell, path + (cell,)))
        all_paths.append(single)
    for combo in itertools.product(*all_paths):
        if not detect_conflicts(tuple(agents), list(combo)) and physically_sound(list(combo)):
            return list(combo)
    return None


def random_instance(rng, max_agents=3, obstacle_chance=0.2, max_side=6):
    width = rng.randint(2, max_side)
    height = rng.randint(2, max_side)
    rows = [
        "".join("#" if rng.random() < obstacle_chance else "." for _ in range(width))
        for _ in range(height)
    ]
    free = [(x, y) for y in range(height) for x in range(width) if rows[y][x] == "."]
    if len(free) < 3:
        return None
    map_rec = Map.parse({"id": "m", "width": width, "height": height, "grid": rows})
    count = rng.randint(1, min(max_agents, len(free) - 1))
    starts = rng.sample(free, count)
    goals = []
    for start in starts:
        canvas = [cell for cell in free if cell != start and cell not in goals]
        if not canvas:
            return None
        goals.append(rng.choice(canvas))
    agents = [
        {"id": f"a{index}", "start": list(start), "goal": list(goal)}
        for index, (start, goal) in enumerate(zip(starts, goals))
    ]
    mission = Mission.parse({"id": "x", "map_id": "m", "agents": agents}, map_rec)
    return map_rec, mission


def main():
    rng = random.Random(20260101)
    solved = 0
    unsolvable = 0
    methods = {"priority": 0, "constraint_tree": 0}
    trials = 0
    while trials < 800:
        instance = random_instance(rng)
        if instance is None:
            continue
        trials += 1
        map_rec, mission = instance
        result = plan_mission(map_rec, mission)
        if result.status == "solved":
            assert_solved(map_rec, mission, result)
            methods[result.method] += 1
            solved += 1
        else:
            assert not result.paths
            assert result.conflicts
            unsolvable += 1
    print(
        f"random invariants ok: trials={trials} solved={solved} unsolvable={unsolvable} "
        f"methods={methods}"
    )

    rng = random.Random(4242)
    checked = 0
    false_negatives = 0
    witnesses = 0
    trials = 0
    while trials < 600 and checked < 80:
        instance = random_instance(rng, max_agents=2, obstacle_chance=0.3, max_side=4)
        if instance is None:
            continue
        trials += 1
        map_rec, mission = instance
        result = plan_mission(map_rec, mission)
        if result.status == "solved":
            assert_solved(map_rec, mission, result)
            continue
        checked += 1
        witness = brute(map_rec, mission)
        if witness is None:
            continue
        witnesses += 1
        # The witness stays inside a short horizon; ask the planner for a wider
        # budget before calling this a genuine incompleteness bug.
        wider = plan_mission(map_rec, mission, order_budget=5040, tree_budget=4096)
        if wider.status != "solved":
            false_negatives += 1
            print("FALSE NEGATIVE", map_rec.rows, [a.as_dict() for a in mission.agents], witness)
    print(
        f"brute-force compared: {checked} solvable-witness={witnesses} "
        f"false negatives: {false_negatives}"
    )

    instance = random_instance(random.Random(7), max_agents=3)
    map_rec, mission = instance
    first = plan_mission(map_rec, mission)
    second = plan_mission(map_rec, mission)
    assert first == second, "planner is not deterministic"
    print("determinism ok:", first.status, first.method, first.solution_makespan)


if __name__ == "__main__":
    main()
