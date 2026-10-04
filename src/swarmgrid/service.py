from __future__ import annotations

from typing import Any, Callable

from .errors import ConflictError, NotFoundError, ValidationError
from .model import MAX_TICK, Agent, Closure, Map, Mission, Point, identifier, integer, parse_closures, point
from .solver import (
    Closures,
    DEFAULT_ORDER_BUDGET,
    DEFAULT_TREE_BUDGET,
    ReplanResult,
    detect_conflicts,
    plan_mission,
    position_at,
    replan_mission,
)
from .store import Store

MAX_ORDER_BUDGET = 5040
MAX_TREE_BUDGET = 4096


class SwarmGrid:
    """MAPF scheduling service: maps in, missions in, conflict-free joint plans out."""

    def __init__(self, database: str):
        self.store = Store(database)
        self._maps: dict[str, Map] = {}

    # ------------------------------------------------------------------ #
    # maps
    # ------------------------------------------------------------------ #

    def create_map(self, raw: Any, key: str | None) -> dict[str, Any]:
        record = Map.parse(raw)

        def create() -> dict[str, Any]:
            existing = self.store.connection.execute(
                "SELECT 1 FROM maps WHERE id = ?", (record.id,)
            ).fetchone()
            if existing:
                raise ConflictError(f"map {record.id} already exists")
            self.store.connection.execute(
                "INSERT INTO maps(id, document) VALUES (?, ?)",
                (record.id, self.store.encode(record.as_dict())),
            )
            return record.as_dict()

        return self._idempotent(key, f"create-map:{record.id}", create)

    def get_map(self, map_id: str) -> dict[str, Any]:
        return self._load_map(map_id).as_dict()

    def _load_map(self, map_id: str) -> Map:
        cached = self._maps.get(map_id)
        if cached is not None:
            return cached
        row = self.store.connection.execute(
            "SELECT document FROM maps WHERE id = ?", (map_id,)
        ).fetchone()
        if not row:
            raise NotFoundError(f"map {map_id} was not found")
        record = Map.parse(self.store.decode(row["document"]))
        self._maps[map_id] = record
        return record

    # ------------------------------------------------------------------ #
    # missions
    # ------------------------------------------------------------------ #

    def create_mission(self, raw: Any, key: str | None) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValidationError("mission must be an object")
        unknown = sorted(set(raw) - {"id", "map_id", "agents"})
        if unknown:
            raise ValidationError(f"mission contains unknown fields: {', '.join(unknown)}")
        for field in ("id", "map_id", "agents"):
            if field not in raw:
                raise ValidationError(f"mission is missing required field: {field}")
        mission_id = identifier(raw["id"], "mission id")
        map_id = identifier(raw["map_id"], "map_id")
        map_record = self._load_map(map_id)
        mission = Mission.parse(raw, map_record)

        def create() -> dict[str, Any]:
            existing = self.store.connection.execute(
                "SELECT 1 FROM missions WHERE id = ?", (mission_id,)
            ).fetchone()
            if existing:
                raise ConflictError(f"mission {mission_id} already exists")
            state = {
                "id": mission_id,
                "map_id": map_id,
                "status": "created",
                "created_at": self.store.now(),
                "agents": [agent.as_dict() for agent in mission.agents],
            }
            self.store.connection.execute(
                "INSERT INTO missions(id, map_id, state, plan) VALUES (?, ?, ?, NULL)",
                (mission_id, map_id, self.store.encode(state)),
            )
            return state

        return self._idempotent(key, f"create-mission:{mission_id}", create)

    def get_mission(self, mission_id: str) -> dict[str, Any]:
        state, plan = self._load_mission(mission_id)
        view = dict(state)
        view["plan"] = plan
        return view

    def _load_mission(self, mission_id: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        row = self.store.connection.execute(
            "SELECT state, plan FROM missions WHERE id = ?", (mission_id,)
        ).fetchone()
        if not row:
            raise NotFoundError(f"mission {mission_id} was not found")
        state = self.store.decode(row["state"])
        plan = self.store.decode(row["plan"]) if row["plan"] is not None else None
        return state, plan

    def plan(self, mission_id: str, raw: Any, key: str | None) -> dict[str, Any]:
        """Plan `mission_id`, returning the stored plan when it was already solved."""

        def resolve() -> dict[str, Any]:
            state, stored_plan = self._load_mission(mission_id)
            if state["status"] == "solved" and stored_plan is not None:
                return stored_plan

            map_record = self._load_map(state["map_id"])
            mission = self._rebuild_mission(state)
            requested, order_budget, tree_budget, closures = self._plan_options(
                raw, mission, map_record
            )
            result = plan_mission(
                map_record, mission, requested, order_budget, tree_budget, closures
            )
            plan = self._materialize(mission, result)
            self._assert_plan_invariants(mission, plan, closures)

            if result.status == "solved":
                state["status"] = "solved"
                state["planned_at"] = self.store.now()
                self.store.connection.execute(
                    "UPDATE missions SET state = ?, plan = ? WHERE id = ?",
                    (self.store.encode(state), self.store.encode(plan), mission_id),
                )
            else:
                state["status"] = "unsolvable"
                state["planned_at"] = self.store.now()
                self.store.connection.execute(
                    "UPDATE missions SET state = ?, plan = NULL WHERE id = ?",
                    (self.store.encode(state), mission_id),
                )
            return plan

        return self._idempotent(key, f"plan-mission:{mission_id}", resolve)

    def schedule(self, mission_id: str) -> dict[str, Any]:
        """Timeline of the solved plan: one frame per timestep from 0 to makespan."""
        state, plan = self._load_mission(mission_id)
        if state["status"] != "solved" or plan is None:
            raise ConflictError(f"mission {mission_id} has no solved plan to schedule")
        paths = {
            entry["agent_id"]: [(int(cell[0]), int(cell[1])) for cell in entry["path"]]
            for entry in plan["agents"]
        }
        return {
            "mission_id": mission_id,
            "map_id": state["map_id"],
            "makespan": plan["makespan"],
            "priority_order": plan["priority_order"],
            "timeline": [
                {
                    "timestep": timestep,
                    "positions": {
                        agent_id: (
                            list(position) if (position := position_at(path, timestep)) else None
                        )
                        for agent_id, path in sorted(paths.items())
                    },
                }
                for timestep in range(plan["makespan"] + 1)
            ],
        }

    def replan(self, mission_id: str, raw: Any, key: str | None) -> dict[str, Any]:
        """Replan `mission_id` from an execution snapshot.

        The stored plan, mission state, and schedule are never touched: the
        result is a stand-alone continuation of the saved solved plan from
        `observed_tick` on, memoized only under the idempotency key.
        """

        def resolve() -> dict[str, Any]:
            state, stored_plan = self._load_mission(mission_id)
            if state["status"] != "solved" or stored_plan is None:
                raise ConflictError(f"mission {mission_id} has no solved plan to replan from")
            map_record = self._load_map(state["map_id"])
            mission = self._rebuild_mission(state)
            observed_tick, positions, order_budget, tree_budget, closures = self._replan_options(
                raw, mission, map_record
            )
            result = replan_mission(
                map_record, mission, positions, observed_tick, order_budget, tree_budget, closures
            )
            response = self._materialize_replan(mission, positions, observed_tick, result)
            self._assert_replan_invariants(mission, positions, observed_tick, response, closures)
            return response

        return self._idempotent(key, f"replan-mission:{mission_id}", resolve)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _plan_options(self, raw: Any, mission: Mission, map_record: Map) -> tuple[
        list[str] | None, int, int, Closures
    ]:
        if raw is None:
            return None, DEFAULT_ORDER_BUDGET, DEFAULT_TREE_BUDGET, ()
        if not isinstance(raw, dict):
            raise ValidationError("plan body must be an object")
        unknown = sorted(set(raw) - {"priority", "order_budget", "tree_budget", "closures"})
        if unknown:
            raise ValidationError(f"plan body contains unknown fields: {', '.join(unknown)}")
        known = sorted(agent.id for agent in mission.agents)
        priority = raw.get("priority")
        if priority is not None:
            if not isinstance(priority, list) or any(
                not isinstance(agent_id, str) or not agent_id for agent_id in priority
            ):
                raise ValidationError("priority must be an array of non-empty agent id strings")
            if len(set(priority)) != len(priority):
                raise ValidationError("priority must not repeat an agent id")
            if sorted(priority) != known:
                raise ValidationError("priority must list every agent of the mission exactly once")
        budget = raw.get("order_budget", DEFAULT_ORDER_BUDGET)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1 or budget > MAX_ORDER_BUDGET:
            raise ValidationError(f"order_budget must be an integer between 1 and {MAX_ORDER_BUDGET}")
        tree_budget = raw.get("tree_budget", DEFAULT_TREE_BUDGET)
        if (
            isinstance(tree_budget, bool)
            or not isinstance(tree_budget, int)
            or tree_budget < 0
            or tree_budget > MAX_TREE_BUDGET
        ):
            raise ValidationError(f"tree_budget must be an integer between 0 and {MAX_TREE_BUDGET}")
        closures = parse_closures(raw.get("closures"), map_record)
        return (
            list(priority) if priority is not None else None
        ), budget, tree_budget, closures

    def _replan_options(
        self, raw: Any, mission: Mission, map_record: Map
    ) -> tuple[int, dict[str, Point], int, int, Closures]:
        if not isinstance(raw, dict):
            raise ValidationError("replan body must be an object")
        unknown = sorted(
            set(raw) - {"observed_tick", "positions", "closures", "order_budget", "tree_budget"}
        )
        if unknown:
            raise ValidationError(f"replan body contains unknown fields: {', '.join(unknown)}")
        for field in ("observed_tick", "positions"):
            if field not in raw:
                raise ValidationError(f"replan body is missing required field: {field}")
        observed_tick = integer(raw["observed_tick"], "observed_tick", 0, MAX_TICK)
        positions = self._snapshot_positions(raw["positions"], mission, map_record)
        order_budget = raw.get("order_budget", DEFAULT_ORDER_BUDGET)
        if (
            isinstance(order_budget, bool)
            or not isinstance(order_budget, int)
            or order_budget < 1
            or order_budget > MAX_ORDER_BUDGET
        ):
            raise ValidationError(
                f"order_budget must be an integer between 1 and {MAX_ORDER_BUDGET}"
            )
        tree_budget = raw.get("tree_budget", DEFAULT_TREE_BUDGET)
        if (
            isinstance(tree_budget, bool)
            or not isinstance(tree_budget, int)
            or tree_budget < 0
            or tree_budget > MAX_TREE_BUDGET
        ):
            raise ValidationError(
                f"tree_budget must be an integer between 0 and {MAX_TREE_BUDGET}"
            )
        closures = parse_closures(raw.get("closures"), map_record)
        for index, closure in enumerate(closures):
            if closure.start_tick < observed_tick:
                raise ValidationError(
                    f"closures[{index}].start_tick must not be earlier than observed_tick"
                )
        return observed_tick, positions, order_budget, tree_budget, closures

    def _snapshot_positions(
        self, raw: Any, mission: Mission, map_record: Map
    ) -> dict[str, Point]:
        if not isinstance(raw, dict):
            raise ValidationError("positions must be an object keyed by agent id")
        known = {agent.id for agent in mission.agents}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValidationError(f"positions contains unknown agents: {', '.join(unknown)}")
        missing = sorted(known - set(raw))
        if missing:
            raise ValidationError(f"positions is missing agents: {', '.join(missing)}")
        positions: dict[str, Point] = {}
        for agent_id in sorted(raw):
            cell = point(raw[agent_id], f"positions[{agent_id}]", map_record.width, map_record.height)
            if not map_record.walkable(cell):
                raise ValidationError(f"positions[{agent_id}] {list(cell)} is an obstacle cell")
            positions[agent_id] = cell
        if len(set(positions.values())) != len(positions):
            raise ValidationError("positions must place every agent on a distinct cell")
        return positions

    def _rebuild_mission(self, state: dict[str, Any]) -> Mission:
        raw = {
            "id": state["id"],
            "map_id": state["map_id"],
            "agents": [
                {"id": entry["id"], "start": entry["start"], "goal": entry["goal"]}
                for entry in state["agents"]
            ],
        }
        return Mission.parse(raw, self._load_map(state["map_id"]))

    def _materialize(self, mission: Mission, result: Any) -> dict[str, Any]:
        by_id = {agent.id: agent for agent in mission.agents}
        if result.status == "solved":
            agents = [
                {
                    "agent_id": agent_id,
                    "start": list(by_id[agent_id].start),
                    "goal": list(by_id[agent_id].goal),
                    "arrival": len(path) - 1,
                    "path": [list(cell) for cell in path],
                }
                for agent_id, path in zip(result.order, result.paths)
            ]
            return {
                "mission_id": mission.id,
                "map_id": mission.map_id,
                "status": "solved",
                "method": result.method,
                "makespan": result.solution_makespan,
                "priority_order": list(result.order),
                "orders_tried": result.orders_tried,
                "orders_budget": result.orders_budget,
                "agents": sorted(agents, key=lambda entry: entry["agent_id"]),
            }
        return {
            "mission_id": mission.id,
            "map_id": mission.map_id,
            "status": "unsolvable",
            "method": None,
            "makespan": None,
            "priority_order": [],
            "orders_tried": result.orders_tried,
            "orders_budget": result.orders_budget,
            "conflicts": [conflict.as_dict() for conflict in result.conflicts],
        }

    def _assert_plan_invariants(
        self, mission: Mission, plan: dict[str, Any], closures: Closures = ()
    ) -> None:
        """Fail loudly rather than publish a plan that breaks the documented invariants."""
        if plan["status"] != "solved":
            return
        map_record = self._load_map(mission.map_id)
        by_id = {agent.id: agent for agent in mission.agents}
        order: list[Agent] = []
        paths: list[tuple[tuple[int, int], ...]] = []
        for entry in plan["agents"]:
            agent_id = entry["agent_id"]
            path = tuple(tuple(cell) for cell in entry["path"])
            if not path or path[0] != by_id[agent_id].start or path[-1] != by_id[agent_id].goal:
                raise ConflictError("internal planner error: a path does not join start to goal")
            for index, cell in enumerate(path):
                if not map_record.walkable(cell):
                    raise ConflictError("internal planner error: a path crosses an obstacle")
                if index and abs(path[index - 1][0] - cell[0]) + abs(path[index - 1][1] - cell[1]) > 1:
                    raise ConflictError("internal planner error: a path is not continuous")
                for closure in closures:
                    if closure.start_tick <= index <= closure.end_tick and cell in closure.cells_set:
                        raise ConflictError(
                            "internal planner error: a path occupies a temporarily closed cell"
                        )
            if entry["arrival"] != len(path) - 1:
                raise ConflictError("internal planner error: arrival does not match the path length")
            order.append(by_id[agent_id])
            paths.append(path)
        if len({tuple(path[-1]) for path in paths}) != len(paths):
            raise ConflictError("internal planner error: two agents settle on the same goal cell")
        for timestep in range(plan["makespan"] + 1):
            held = [position_at(path, timestep) for path in paths]
            occupied = [cell for cell in held if cell is not None]
            if len(set(occupied)) != len(occupied):
                raise ConflictError("internal planner error: two agents share a cell at one timestep")
        if max(len(path) - 1 for path in paths) != plan["makespan"]:
            raise ConflictError("internal planner error: makespan is not the latest arrival")
        if detect_conflicts(tuple(order), paths):
            raise ConflictError("internal planner error: the published plan still has a conflict")

    def _materialize_replan(
        self,
        mission: Mission,
        positions: dict[str, Point],
        observed_tick: int,
        result: ReplanResult,
    ) -> dict[str, Any]:
        by_id = {agent.id: agent for agent in mission.agents}
        if result.status == "solved":
            agents = [
                {
                    "agent_id": agent_id,
                    "start": list(positions[agent_id]),
                    "goal": list(by_id[agent_id].goal),
                    "arrival": observed_tick + len(result.paths[agent_id]) - 1,
                    "path": [list(cell) for cell in result.paths[agent_id]],
                }
                for agent_id in sorted(result.paths)
            ]
            return {
                "mission_id": mission.id,
                "map_id": mission.map_id,
                "status": "solved",
                "method": result.method,
                "makespan": max(entry["arrival"] for entry in agents),
                "replan_from": observed_tick,
                "priority_order": list(result.order),
                "orders_tried": result.orders_tried,
                "orders_budget": result.orders_budget,
                "agents": agents,
            }
        return {
            "mission_id": mission.id,
            "map_id": mission.map_id,
            "status": "unsolvable",
            "method": None,
            "makespan": None,
            "replan_from": observed_tick,
            "priority_order": [],
            "orders_tried": result.orders_tried,
            "orders_budget": result.orders_budget,
            "conflicts": [conflict.as_dict() for conflict in result.conflicts],
        }

    def _assert_replan_invariants(
        self,
        mission: Mission,
        positions: dict[str, Point],
        observed_tick: int,
        response: dict[str, Any],
        closures: Closures = (),
    ) -> None:
        """The plan invariants, rebased so path index `i` is tick `observed_tick + i`."""
        if response["status"] != "solved":
            return
        map_record = self._load_map(mission.map_id)
        by_id = {agent.id: agent for agent in mission.agents}
        paths: dict[str, tuple[Point, ...]] = {}
        for entry in response["agents"]:
            agent_id = entry["agent_id"]
            path = tuple(tuple(cell) for cell in entry["path"])
            if not path or path[0] != positions[agent_id] or path[-1] != by_id[agent_id].goal:
                raise ConflictError(
                    "internal planner error: a path does not join the snapshot to the goal"
                )
            for index, cell in enumerate(path):
                if not map_record.walkable(cell):
                    raise ConflictError("internal planner error: a path crosses an obstacle")
                if index and abs(path[index - 1][0] - cell[0]) + abs(path[index - 1][1] - cell[1]) > 1:
                    raise ConflictError("internal planner error: a path is not continuous")
                tick = observed_tick + index
                for closure in closures:
                    if closure.start_tick <= tick <= closure.end_tick and cell in closure.cells_set:
                        raise ConflictError(
                            "internal planner error: a path occupies a temporarily closed cell"
                        )
            if entry["arrival"] != observed_tick + len(path) - 1:
                raise ConflictError("internal planner error: arrival does not match the path length")
            paths[agent_id] = path
        if len({path[-1] for path in paths.values()}) != len(paths):
            raise ConflictError("internal planner error: two agents settle on the same goal cell")
        ordered = tuple(by_id[agent_id] for agent_id in sorted(paths))
        if detect_conflicts(ordered, [paths[agent.id] for agent in ordered]):
            raise ConflictError("internal planner error: the published plan still has a conflict")
        if response["makespan"] != max(observed_tick + len(path) - 1 for path in paths.values()):
            raise ConflictError("internal planner error: makespan is not the latest arrival")

    def _idempotent(
        self, key: str | None, operation: str, action: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        if not key:
            raise ValidationError("Idempotency-Key header is required")
        with self.store.transaction() as connection:
            existing = connection.execute(
                "SELECT operation, response FROM idempotency WHERE key = ?", (key,)
            ).fetchone()
            if existing:
                if existing["operation"] != operation:
                    raise ConflictError("idempotency key was already used for another operation")
                return self.store.decode(existing["response"])
            response = action()
            # Store and return the same encoding so a replayed request is
            # byte-identical to the first response.
            encoded = self.store.encode(response)
            connection.execute(
                "INSERT INTO idempotency(key, operation, response) VALUES (?, ?, ?)",
                (key, operation, encoded),
            )
            return self.store.decode(encoded)


__all__ = ["SwarmGrid", "Conflict"]
