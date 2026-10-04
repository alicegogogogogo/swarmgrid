import tempfile
import unittest
from pathlib import Path

from swarmgrid.errors import ConflictError, NotFoundError, ValidationError
from swarmgrid.service import SwarmGrid


def corridor_map(map_id: str = "corridor") -> dict:
    """A 5x3 room split by an obstacle wall with a single gap in the middle."""
    return {
        "id": map_id,
        "width": 5,
        "height": 3,
        "grid": [
            ".....",
            "..##.",
            ".....",
        ],
    }


class MapValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.service = SwarmGrid(str(Path(self.directory.name) / "test.db"))

    def tearDown(self):
        self.directory.cleanup()

    def test_create_map_returns_normalized_document(self):
        created = self.service.create_map(corridor_map(), "map-1")
        self.assertEqual(
            {
                "id": "corridor",
                "width": 5,
                "height": 3,
                "grid": [".....", "..##.", "....."],
                "obstacle": "#",
            },
            created,
        )
        self.assertEqual(created, self.service.get_map("corridor"))

    def test_custom_obstacle_character(self):
        created = self.service.create_map(
            {"id": "custom", "width": 3, "height": 1, "grid": ["x.."], "obstacle": "x"}, "k1"
        )
        self.assertEqual("x", created["obstacle"])

    def test_unknown_map_field_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "unknown fields"):
            self.service.create_map({**corridor_map(), "notes": "hi"}, "k1")

    def test_grid_shape_is_validated(self):
        base = {"id": "m", "width": 3, "height": 2}
        cases = [
            ({**base, "grid": ["..."]}, "exactly 2 row"),
            ({**base, "grid": ["...", ".."]}, "exactly 3 characters"),
            ({**base, "grid": ["...", "..o"], "obstacle": "@"}, "must be '.' or the obstacle"),
            ({**base, "grid": ["...", 5]}, "must be a string"),
        ]
        for index, (payload, pattern) in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValidationError, pattern):
                    self.service.create_map(payload, f"k{index}")

    def test_dimension_bounds_are_validated(self):
        def payload(width, height):
            return {"id": "m", "width": width, "height": height, "grid": ["." * width] * height}

        def rejected(payload, pattern, key):
            with self.assertRaisesRegex(ValidationError, pattern):
                self.service.create_map(payload, key)

        rejected(payload(0, 1), "between 1 and 64", "k1")
        rejected(payload(3, 0), "between 1 and 64", "k2")
        rejected(payload(65, 1), "between 1 and 64", "k3")
        rejected({"id": "m", "width": 3.0, "height": 1, "grid": ["..."]}, "must be an integer", "k4")
        self.assertEqual(64, len(self.service.create_map(payload(64, 64), "k5")["grid"]))

    def test_duplicate_map_is_a_conflict(self):
        self.service.create_map(corridor_map(), "k1")
        with self.assertRaisesRegex(ConflictError, "already exists"):
            self.service.create_map(corridor_map(), "k2")

    def test_missing_map_is_not_found(self):
        with self.assertRaises(NotFoundError):
            self.service.get_map("nope")

    def test_idempotency_key_is_required_and_replayable(self):
        with self.assertRaisesRegex(ValidationError, "Idempotency-Key"):
            self.service.create_map(corridor_map(), None)
        first = self.service.create_map(corridor_map(), "same")
        self.assertEqual(first, self.service.create_map(corridor_map(), "same"))

    def test_idempotency_key_cannot_cross_operations(self):
        self.service.create_map(corridor_map(), "shared")
        with self.assertRaises(ConflictError):
            self.service.create_mission(
                {
                    "id": "m1",
                    "map_id": "corridor",
                    "agents": [{"id": "a", "start": [0, 0], "goal": [4, 0]}],
                },
                "shared",
            )


class MissionValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.service = SwarmGrid(str(Path(self.directory.name) / "test.db"))
        self.service.create_map(corridor_map(), "map-1")

    def tearDown(self):
        self.directory.cleanup()

    def mission(self, agents, mission_id="m1"):
        return {"id": mission_id, "map_id": "corridor", "agents": agents}

    def test_create_mission_starts_in_created_state(self):
        created = self.service.create_mission(
            self.mission([{"id": "a", "start": [0, 0], "goal": [4, 0]}]), "k1"
        )
        self.assertEqual("created", created["status"])
        self.assertIsNone(self.service.get_mission("m1")["plan"])

    def test_unknown_mission_field_is_rejected(self):
        payload = {**self.mission([{"id": "a", "start": [0, 0], "goal": [4, 0]}]), "priority": []}
        with self.assertRaisesRegex(ValidationError, "unknown fields"):
            self.service.create_mission(payload, "k1")

    def test_unknown_agent_field_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "agents\\[0\\] contains unknown fields"):
            self.service.create_mission(
                self.mission([{"id": "a", "start": [0, 0], "goal": [4, 0], "speed": 2}]), "k1"
            )

    def test_map_must_exist(self):
        with self.assertRaises(NotFoundError):
            self.service.create_mission(
                {"id": "m1", "map_id": "missing", "agents": [{"id": "a", "start": [0, 0], "goal": [4, 0]}]},
                "k1",
            )

    def test_agents_must_be_unique_and_distinguishable(self):
        with self.assertRaisesRegex(ValidationError, "unique"):
            self.service.create_mission(
                self.mission(
                    [
                        {"id": "a", "start": [0, 0], "goal": [4, 0]},
                        {"id": "a", "start": [0, 2], "goal": [4, 2]},
                    ]
                ),
                "k1",
            )
        with self.assertRaisesRegex(ValidationError, "start cells must be unique"):
            self.service.create_mission(
                self.mission(
                    [
                        {"id": "a", "start": [0, 0], "goal": [4, 0]},
                        {"id": "b", "start": [0, 0], "goal": [4, 2]},
                    ]
                ),
                "k2",
            )
        with self.assertRaisesRegex(ValidationError, "already starts on its goal"):
            self.service.create_mission(
                self.mission([{"id": "a", "start": [0, 0], "goal": [0, 0]}]), "k3"
            )

    def test_cells_must_be_in_bounds_and_free(self):
        with self.assertRaisesRegex(ValidationError, "between 0 and 4"):
            self.service.create_mission(
                self.mission([{"id": "a", "start": [5, 0], "goal": [4, 0]}]), "k1"
            )
        with self.assertRaisesRegex(ValidationError, "obstacle"):
            self.service.create_mission(
                self.mission([{"id": "a", "start": [0, 0], "goal": [2, 1]}]), "k2"
            )
        with self.assertRaisesRegex(ValidationError, r"\[x, y\] pair"):
            self.service.create_mission(
                self.mission([{"id": "a", "start": [0], "goal": [4, 0]}]), "k3"
            )

    def test_duplicate_mission_is_a_conflict(self):
        payload = self.mission([{"id": "a", "start": [0, 0], "goal": [4, 0]}])
        self.service.create_mission(payload, "k1")
        with self.assertRaisesRegex(ConflictError, "already exists"):
            self.service.create_mission(payload, "k2")


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.service = SwarmGrid(str(Path(self.directory.name) / "test.db"))
        self.service.create_map(corridor_map(), "map-1")

    def tearDown(self):
        self.directory.cleanup()

    def create(self, agents, mission_id="m1"):
        self.service.create_mission(
            {"id": mission_id, "map_id": "corridor", "agents": agents}, f"mission-{mission_id}"
        )

    def test_single_agent_takes_the_shortest_route(self):
        self.create([{"id": "a", "start": [0, 0], "goal": [4, 0]}])
        plan = self.service.plan("m1", {}, "p1")
        self.assertEqual("solved", plan["status"])
        self.assertEqual(4, plan["makespan"])
        self.assertEqual(["a"], plan["priority_order"])
        self.assertEqual(
            [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], plan["agents"][0]["path"]
        )
        self.assertEqual(4, plan["agents"][0]["arrival"])

    def test_crossing_agents_are_resolved_by_replanning(self):
        self.create(
            [
                {"id": "a", "start": [0, 0], "goal": [4, 0]},
                {"id": "b", "start": [4, 0], "goal": [0, 0]},
            ]
        )
        plan = self.service.plan("m1", {}, "p1")
        self.assertEqual("solved", plan["status"])
        self.assertEqual("priority", plan["method"])
        self.assertEqual(["a", "b"], plan["priority_order"])
        self.assertEqual(1, plan["orders_tried"])
        # a takes the direct lane, so b yields and detours through the lower
        # lane instead of swapping with a.
        self.assertEqual(4, plan["agents"][0]["arrival"])
        self.assertEqual(8, plan["agents"][1]["arrival"])
        self.assertEqual(
            [[4, 0], [4, 1], [4, 2], [3, 2], [2, 2], [1, 2], [0, 2], [0, 1], [0, 0]],
            plan["agents"][1]["path"],
        )

    def test_plan_invariants_hold_for_every_published_plan(self):
        open_map = {
            "id": "open",
            "width": 6,
            "height": 4,
            "grid": ["......", "......", "......", "......"],
        }
        self.service.create_map(open_map, "map-open")
        self.service.create_mission(
            {
                "id": "fleet",
                "map_id": "open",
                "agents": [
                    {"id": "a", "start": [0, 0], "goal": [5, 0]},
                    {"id": "b", "start": [5, 3], "goal": [0, 3]},
                    {"id": "c", "start": [2, 1], "goal": [2, 2]},
                ],
            },
            "mission-fleet",
        )
        plan = self.service.plan("fleet", {}, "p1")
        self.assertEqual("solved", plan["status"])
        map_record = self.service.get_map("open")
        trajectories: dict[str, list[tuple]] = {}
        for entry in plan["agents"]:
            path = [tuple(cell) for cell in entry["path"]]
            self.assertEqual(tuple(entry["start"]), path[0])
            self.assertEqual(tuple(entry["goal"]), path[-1])
            self.assertEqual(len(path) - 1, entry["arrival"])
            for index, cell in enumerate(path):
                self.assertNotEqual("#", map_record["grid"][cell[1]][cell[0]])
                if index:
                    previous = path[index - 1]
                    self.assertLessEqual(
                        abs(previous[0] - cell[0]) + abs(previous[1] - cell[1]), 1
                    )
            trajectories[entry["agent_id"]] = path
        for timestep in range(plan["makespan"] + 1):
            occupied = [
                path[timestep]
                for path in trajectories.values()
                if timestep < len(path)
            ]
            self.assertEqual(len(occupied), len(set(occupied)), f"collision at t={timestep}")
        self.assertEqual(plan["makespan"], max(entry["arrival"] for entry in plan["agents"]))
        self.assertEqual(3, len({tuple(path[-1]) for path in trajectories.values()}))

    def test_priority_order_can_be_pinned_explicitly(self):
        square = {"id": "square", "width": 2, "height": 2, "grid": ["..", ".."]}
        self.service.create_map(square, "map-square")
        self.service.create_mission(
            {
                "id": "corner",
                "map_id": "square",
                "agents": [
                    {"id": "agv-1", "start": [0, 0], "goal": [1, 1]},
                    {"id": "agv-2", "start": [1, 0], "goal": [0, 0]},
                ],
            },
            "mission-corner",
        )
        pinned = self.service.plan("corner", {"priority": ["agv-2", "agv-1"]}, "p-pinned")
        self.assertEqual("solved", pinned["status"])
        self.assertEqual(["agv-2", "agv-1"], pinned["priority_order"])
        self.assertEqual(1, pinned["orders_budget"])
        paths = {entry["agent_id"]: entry["path"] for entry in pinned["agents"]}
        self.assertEqual([[1, 0], [0, 0]], paths["agv-2"])
        self.assertEqual(
            [[0, 0], [0, 1], [1, 1]], paths["agv-1"]
        )
        self.assertEqual(2, pinned["makespan"])
        self.assertEqual(pinned["makespan"], max(entry["arrival"] for entry in pinned["agents"]))
        for frame in self.service.schedule("corner")["timeline"]:
            live = [tuple(value) for value in frame["positions"].values() if value is not None]
            self.assertEqual(len(live), len(set(live)))

    def test_head_on_agents_in_a_single_lane_are_unsolvable(self):
        narrow = {
            "id": "narrow",
            "width": 4,
            "height": 1,
            "grid": ["...."],
        }
        self.service.create_map(narrow, "map-narrow")
        self.service.create_mission(
            {
                "id": "swap",
                "map_id": "narrow",
                "agents": [
                    {"id": "a", "start": [0, 0], "goal": [3, 0]},
                    {"id": "b", "start": [3, 0], "goal": [0, 0]},
                ],
            },
            "mission-swap",
        )
        plan = self.service.plan("swap", {}, "p1")
        self.assertEqual("unsolvable", plan["status"])
        self.assertIsNone(plan["makespan"])
        self.assertEqual([], plan["priority_order"])
        self.assertTrue(plan["conflicts"])
        kinds = {conflict["type"] for conflict in plan["conflicts"]}
        self.assertTrue(kinds <= {"vertex", "edge", "unreachable_goal"})
        self.assertEqual(2, plan["orders_tried"])
        self.assertEqual(2, plan["orders_budget"])
        state = self.service.get_mission("swap")
        self.assertEqual("unsolvable", state["status"])
        self.assertIsNone(state["plan"])

    def test_unreachable_goal_is_reported_without_planning(self):
        walled = {"id": "walled", "width": 5, "height": 1, "grid": ["..#.."]}
        self.service.create_map(walled, "map-walled")
        self.service.create_mission(
            {
                "id": "sealed",
                "map_id": "walled",
                "agents": [{"id": "a", "start": [0, 0], "goal": [4, 0]}],
            },
            "mission-walled",
        )
        plan = self.service.plan("sealed", {}, "p1")
        self.assertEqual("unsolvable", plan["status"])
        self.assertEqual(0, plan["orders_tried"])
        self.assertEqual(["unreachable_goal"], [conflict["type"] for conflict in plan["conflicts"]])
        self.assertEqual([[4, 0]], [plan["conflicts"][0]["position"]])

    def test_unreachable_start_is_unsolvable(self):
        # (0, 0) and (0, 1) sit in a sealed pocket; the goal is outside it.
        sealer = {"id": "sealer", "width": 5, "height": 3, "grid": [".#...", "##...", "....."]}
        self.service.create_map(sealer, "map-sealer")
        self.service.create_mission(
            {
                "id": "sealed",
                "map_id": "sealer",
                "agents": [{"id": "a", "start": [0, 0], "goal": [0, 2]}],
            },
            "mission-sealed",
        )
        plan = self.service.plan("sealed", {}, "p1")
        self.assertEqual("unsolvable", plan["status"])
        self.assertEqual("unreachable_goal", plan["conflicts"][0]["type"])
        self.assertEqual("a", plan["conflicts"][0]["agent"])
        self.assertEqual([[0, 2]], [plan["conflicts"][0]["position"]])

    def test_explicit_priority_order_is_honoured(self):
        self.create(
            [
                {"id": "a", "start": [0, 0], "goal": [4, 0]},
                {"id": "b", "start": [4, 0], "goal": [0, 0]},
            ]
        )
        plan = self.service.plan("m1", {"priority": ["b", "a"]}, "p1")
        self.assertEqual("solved", plan["status"])
        self.assertEqual(["b", "a"], plan["priority_order"])
        self.assertEqual(1, plan["orders_budget"])

    def test_priority_must_cover_the_mission_agents(self):
        self.create(
            [
                {"id": "a", "start": [0, 0], "goal": [4, 0]},
                {"id": "b", "start": [4, 0], "goal": [0, 0]},
            ]
        )
        for priority, pattern in (
            (["a"], "exactly once"),
            (["a", "a"], "must not repeat"),
            (["a", "z"], "exactly once"),
            (["a", 7], "non-empty agent id strings"),
        ):
            with self.subTest(priority=priority):
                with self.assertRaisesRegex(ValidationError, pattern):
                    self.service.plan("m1", {"priority": priority}, "px")

    def test_plan_options_are_validated(self):
        self.create([{"id": "a", "start": [0, 0], "goal": [4, 0]}])
        with self.assertRaisesRegex(ValidationError, "unknown fields"):
            self.service.plan("m1", {"order": ["a"]}, "p1")
        with self.assertRaisesRegex(ValidationError, "order_budget"):
            self.service.plan("m1", {"order_budget": 0}, "p2")
        with self.assertRaisesRegex(ValidationError, "order_budget"):
            self.service.plan("m1", {"order_budget": True}, "p3")
        with self.assertRaisesRegex(ValidationError, "Idempotency-Key"):
            self.service.plan("m1", {}, None)

    def test_planning_missing_mission_is_not_found(self):
        with self.assertRaises(NotFoundError):
            self.service.plan("ghost", {}, "p1")

    def test_repeated_planning_returns_the_recorded_plan(self):
        self.create([{"id": "a", "start": [0, 0], "goal": [4, 0]}])
        first = self.service.plan("m1", {}, "k-plan")
        repeated = self.service.plan("m1", {}, "k-plan")
        again = self.service.plan("m1", {"priority": ["a"]}, "k-plan-2")
        self.assertEqual(first, repeated)
        self.assertEqual(first, again)

    def test_schedule_exposes_one_frame_per_timestep(self):
        self.create(
            [
                {"id": "a", "start": [0, 0], "goal": [1, 0]},
                {"id": "b", "start": [4, 0], "goal": [0, 0]},
            ]
        )
        plan = self.service.plan("m1", {}, "p1")
        schedule = self.service.schedule("m1")
        self.assertEqual(plan["makespan"], schedule["makespan"])
        self.assertEqual(plan["makespan"] + 1, len(schedule["timeline"]))
        self.assertEqual([0, 1], [frame["timestep"] for frame in schedule["timeline"][:2]])
        for frame in schedule["timeline"]:
            self.assertEqual({"a", "b"}, set(frame["positions"]))
        self.assertEqual({"a": [0, 0], "b": [4, 0]}, schedule["timeline"][0]["positions"])
        # a finishes on its goal after one tick, so later frames report null for it.
        self.assertEqual({"a": None, "b": [0, 0]}, schedule["timeline"][-1]["positions"])
        for frame in schedule["timeline"]:
            live = [tuple(value) for value in frame["positions"].values() if value is not None]
            self.assertEqual(len(live), len(set(live)))

    def test_schedule_requires_a_solved_plan(self):
        self.create([{"id": "a", "start": [0, 0], "goal": [4, 0]}])
        with self.assertRaisesRegex(ConflictError, "no solved plan"):
            self.service.schedule("m1")

    def test_schedule_of_an_unsolvable_mission_conflicts(self):
        narrow = {"id": "narrow", "width": 4, "height": 1, "grid": ["...."]}
        self.service.create_map(narrow, "map-narrow")
        self.service.create_mission(
            {
                "id": "swap",
                "map_id": "narrow",
                "agents": [
                    {"id": "a", "start": [0, 0], "goal": [3, 0]},
                    {"id": "b", "start": [3, 0], "goal": [0, 0]},
                ],
            },
            "mission-swap",
        )
        self.service.plan("swap", {}, "p1")
        with self.assertRaisesRegex(ConflictError, "no solved plan"):
            self.service.schedule("swap")

    def test_plans_survive_a_service_restart(self):
        path = str(Path(self.directory.name) / "restart.db")
        first = SwarmGrid(path)
        first.create_map(corridor_map(), "map-1")
        first.create_mission(
            {
                "id": "m1",
                "map_id": "corridor",
                "agents": [{"id": "a", "start": [0, 0], "goal": [4, 0]}],
            },
            "k1",
        )
        plan = first.plan("m1", {}, "k2")
        second = SwarmGrid(path)
        self.assertEqual(plan, second.plan("m1", {}, "k3"))
        self.assertEqual(plan["makespan"], second.schedule("m1")["makespan"])


class ClosurePlanningTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.service = SwarmGrid(str(Path(self.directory.name) / "test.db"))
        self.service.create_map(
            {"id": "lane", "width": 5, "height": 1, "grid": ["....."]}, "map-lane"
        )

    def tearDown(self):
        self.directory.cleanup()

    def create(self, agents, mission_id="m1"):
        self.service.create_mission(
            {"id": mission_id, "map_id": "lane", "agents": agents}, f"mission-{mission_id}"
        )

    def test_closure_forces_a_wait_and_is_honoured_in_the_path(self):
        self.create([{"id": "a", "start": [0, 0], "goal": [4, 0]}])
        plan = self.service.plan(
            "m1", {"closures": [{"cells": [[1, 0]], "start_tick": 1, "end_tick": 1}]}, "p1"
        )
        self.assertEqual("solved", plan["status"])
        self.assertEqual(5, plan["makespan"])
        self.assertEqual(
            [[0, 0], [0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
            plan["agents"][0]["path"],
        )

    def test_permanent_closure_returns_a_closure_conflict(self):
        self.create([{"id": "a", "start": [0, 0], "goal": [4, 0]}])
        plan = self.service.plan(
            "m1",
            {"closures": [{"cells": [[1, 0], [2, 0], [3, 0]], "start_tick": 0, "end_tick": 100000}]},
            "p1",
        )
        self.assertEqual("unsolvable", plan["status"])
        self.assertIsNone(plan["makespan"])
        self.assertNotIn("agents", plan)
        conflict = plan["conflicts"][0]
        self.assertEqual("closure", conflict["type"])
        self.assertEqual("a", conflict["agent"])
        self.assertIsNone(conflict["other"])
        self.assertIsNone(conflict["other_position"])
        self.assertEqual(1, conflict["timestep"])
        self.assertEqual([1, 0], conflict["position"])
        self.assertIn("closure", conflict["message"])

    def test_start_covered_at_zero_returns_one_sorted_diagnostic_per_agent(self):
        self.create(
            [
                {"id": "b", "start": [4, 0], "goal": [0, 0]},
                {"id": "a", "start": [0, 0], "goal": [4, 0]},
            ]
        )
        plan = self.service.plan(
            "m1",
            {"closures": [{"cells": [[0, 0], [4, 0]], "start_tick": 0, "end_tick": 3}]},
            "p1",
        )
        self.assertEqual("unsolvable", plan["status"])
        self.assertEqual(0, plan["orders_tried"])
        self.assertEqual(
            [
                {"agent": "a", "position": [0, 0], "timestep": 0, "type": "closure"},
                {"agent": "b", "position": [4, 0], "timestep": 0, "type": "closure"},
            ],
            [
                {"agent": c["agent"], "position": c["position"], "timestep": c["timestep"], "type": c["type"]}
                for c in plan["conflicts"]
            ],
        )
        for conflict in plan["conflicts"]:
            self.assertIsNone(conflict["other"])
            self.assertIsNone(conflict["other_position"])

    def test_conflicts_are_sorted_and_deduplicated(self):
        self.create(
            [
                {"id": "b", "start": [4, 0], "goal": [0, 0]},
                {"id": "a", "start": [0, 0], "goal": [4, 0]},
            ]
        )
        # Two overlapping closure entries cover both starts; the diagnostics are
        # one per blocked agent regardless of how many entries cover the start.
        body = {
            "closures": [
                {"cells": [[0, 0], [4, 0]], "start_tick": 0, "end_tick": 2},
                {"cells": [[0, 0], [4, 0]], "start_tick": 0, "end_tick": 5},
            ]
        }
        plan = self.service.plan("m1", body, "p1")
        self.assertEqual(["a", "b"], [c["agent"] for c in plan["conflicts"]])

    def test_omitting_closures_matches_the_baseline_plan(self):
        self.create([{"id": "a", "start": [0, 0], "goal": [4, 0]}])
        without = self.service.plan("m1", {}, "p1")
        self.assertEqual([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], without["agents"][0]["path"])
        # An explicit empty closure list on a fresh mission is the same plan.
        self.create([{"id": "a", "start": [0, 0], "goal": [4, 0]}], "m2")
        empty = self.service.plan("m2", {"closures": []}, "p2")
        self.assertEqual(
            [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], empty["agents"][0]["path"]
        )

    def test_closures_are_validated(self):
        self.create([{"id": "a", "start": [0, 0], "goal": [4, 0]}])
        cases = [
            ({"closures": {}}, "closures must be an array"),
            ({"closures": [{}]}, "missing required fields"),
            ({"closures": [{"cells": [], "start_tick": 0, "end_tick": 0}]}, "non-empty"),
            ({"closures": [{"cells": [[1, 0]], "start_tick": 3, "end_tick": 2}]}, "start_tick must not"),
            ({"closures": [{"cells": [[1, 0]], "start_tick": -1, "end_tick": 0}]}, "between 0 and 100000"),
            ({"closures": [{"cells": [[1, 0]], "start_tick": 0, "end_tick": 100001}]}, "between 0 and 100000"),
            ({"closures": [{"cells": [[1, 0], [1, 0]], "start_tick": 0, "end_tick": 0}]}, "must not repeat"),
            ({"closures": [{"cells": [[5, 0]], "start_tick": 0, "end_tick": 0}]}, "between 0 and 4"),
            (
                {"closures": [{"cells": [[1, 0]], "start_tick": 0, "end_tick": 0, "note": 1}]},
                "unknown fields",
            ),
            ({"closures": "nope"}, "closures must be an array"),
            ({"closures": [{"cells": [[1, 0]], "start_tick": True, "end_tick": 0}]}, "must be an integer"),
        ]
        for index, (body, pattern) in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValidationError, pattern):
                    self.service.plan("m1", body, f"bad-{index}")

    def test_closure_array_and_cell_limits_are_enforced(self):
        self.create([{"id": "a", "start": [0, 0], "goal": [4, 0]}])
        with self.assertRaisesRegex(ValidationError, "64 entries"):
            self.service.plan(
                "m1",
                {"closures": [{"cells": [[0, 0]], "start_tick": 0, "end_tick": 0}] * 65},
                "p1",
            )
        with self.assertRaisesRegex(ValidationError, "64 cells"):
            self.service.plan(
                "m1",
                {"closures": [{"cells": [[0, 0]] * 64 + [[1, 0]], "start_tick": 0, "end_tick": 0}]},
                "p2",
            )

    def test_closure_cell_must_not_be_a_static_obstacle(self):
        self.service.create_map(
            {"id": "wall", "width": 3, "height": 1, "grid": [".#."]}, "map-wall"
        )
        self.service.create_mission(
            {"id": "w", "map_id": "wall", "agents": [{"id": "a", "start": [0, 0], "goal": [2, 0]}]},
            "mission-w",
        )
        # The goal itself is statically unreachable here; first confirm the cell
        # is rejected as an obstacle in the closure.
        with self.assertRaisesRegex(ValidationError, "obstacle"):
            self.service.plan(
                "w", {"closures": [{"cells": [[1, 0]], "start_tick": 0, "end_tick": 1}]}, "p1"
            )

    def test_unknown_closure_field_is_rejected(self):
        self.create([{"id": "a", "start": [0, 0], "goal": [4, 0]}])
        with self.assertRaisesRegex(ValidationError, "unknown fields"):
            self.service.plan(
                "m1",
                {"closures": [{"cells": [[0, 0]], "start_tick": 0, "end_tick": 0}], "reason": "x"},
                "p1",
            )

    def test_planning_an_unknown_mission_with_closures_is_not_found(self):
        with self.assertRaises(NotFoundError):
            self.service.plan(
                "ghost", {"closures": [{"cells": [[0, 0]], "start_tick": 0, "end_tick": 1}]}, "p1"
            )

    def test_replayed_closure_plan_returns_the_first_response(self):
        self.create([{"id": "a", "start": [0, 0], "goal": [4, 0]}])
        body = {"closures": [{"cells": [[1, 0]], "start_tick": 1, "end_tick": 1}]}
        first = self.service.plan("m1", body, "same")
        repeated = self.service.plan("m1", body, "same")
        self.assertEqual(first, repeated)
        # A re-plan of the now-solved mission returns the stored plan regardless
        # of the closures in the new body.
        again = self.service.plan("m1", {"closures": []}, "other")
        self.assertEqual(first, again)


class ReplanTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.service = SwarmGrid(str(Path(self.directory.name) / "test.db"))
        self.service.create_map(corridor_map(), "map-1")

    def tearDown(self):
        self.directory.cleanup()

    def create(self, agents, mission_id="m1", map_id="corridor"):
        self.service.create_mission(
            {"id": mission_id, "map_id": map_id, "agents": agents}, f"mission-{mission_id}"
        )

    def create_solved(self, agents=None, mission_id="m1"):
        agents = agents or [
            {"id": "a", "start": [0, 0], "goal": [4, 0]},
            {"id": "b", "start": [4, 0], "goal": [0, 0]},
        ]
        self.create(agents, mission_id)
        return self.service.plan(mission_id, {}, f"plan-{mission_id}")

    def test_replan_continues_from_the_snapshot(self):
        plan = self.create_solved()
        result = self.service.replan(
            "m1", {"observed_tick": 2, "positions": {"a": [2, 0], "b": [4, 2]}}, "r1"
        )
        self.assertEqual("solved", result["status"])
        self.assertEqual(2, result["replan_from"])
        self.assertEqual(plan["mission_id"], result["mission_id"])
        self.assertEqual(["a", "b"], [entry["agent_id"] for entry in result["agents"]])
        paths = {entry["agent_id"]: entry["path"] for entry in result["agents"]}
        self.assertEqual([2, 0], paths["a"][0])
        self.assertEqual([4, 2], paths["b"][0])
        self.assertEqual([4, 0], paths["a"][-1])
        self.assertEqual([0, 0], paths["b"][-1])
        arrivals = {entry["agent_id"]: entry["arrival"] for entry in result["agents"]}
        for agent_id, entry in (("a", paths["a"]), ("b", paths["b"])):
            self.assertEqual(2 + len(entry) - 1, arrivals[agent_id])
        self.assertEqual(result["makespan"], max(arrivals.values()))
        self.assertEqual(["a", "b"], result["priority_order"])
        # No vertex collisions at any absolute tick of the continuation.
        for timestep in range(result["makespan"] + 1 - 2):
            occupied = [
                tuple(path[timestep]) for path in paths.values() if timestep < len(path)
            ]
            self.assertEqual(len(occupied), len(set(occupied)))

    def test_replan_requires_a_solved_plan(self):
        self.create([{"id": "a", "start": [0, 0], "goal": [4, 0]}])
        with self.assertRaisesRegex(ConflictError, "no solved plan"):
            self.service.replan("m1", {"observed_tick": 0, "positions": {"a": [0, 0]}}, "r1")

    def test_replan_of_an_unsolvable_mission_conflicts(self):
        self.service.create_map(
            {"id": "narrow", "width": 4, "height": 1, "grid": ["...."]}, "map-narrow"
        )
        self.create(
            [
                {"id": "a", "start": [0, 0], "goal": [3, 0]},
                {"id": "b", "start": [3, 0], "goal": [0, 0]},
            ],
            "swap",
            "narrow",
        )
        self.assertEqual("unsolvable", self.service.plan("swap", {}, "p1")["status"])
        with self.assertRaisesRegex(ConflictError, "no solved plan"):
            self.service.replan(
                "swap",
                {"observed_tick": 0, "positions": {"a": [0, 0], "b": [3, 0]}},
                "r1",
            )

    def test_replanning_a_missing_mission_is_not_found(self):
        with self.assertRaises(NotFoundError):
            self.service.replan("ghost", {"observed_tick": 0, "positions": {}}, "r1")

    def test_replan_requires_an_idempotency_key(self):
        self.create_solved([{"id": "a", "start": [0, 0], "goal": [4, 0]}])
        with self.assertRaisesRegex(ValidationError, "Idempotency-Key"):
            self.service.replan("m1", {"observed_tick": 0, "positions": {"a": [0, 0]}}, None)

    def test_replan_body_is_validated(self):
        self.create_solved([{"id": "a", "start": [0, 0], "goal": [4, 0]}])
        base = {"observed_tick": 1, "positions": {"a": [1, 0]}}
        cases = [
            ({**base, "priority": ["a"]}, "unknown fields"),
            ({"positions": {"a": [1, 0]}}, "missing required field: observed_tick"),
            ({"observed_tick": 1}, "missing required field: positions"),
            ({**base, "observed_tick": -1}, "observed_tick must be between 0 and 100000"),
            ({**base, "observed_tick": 100001}, "observed_tick must be between 0 and 100000"),
            ({**base, "observed_tick": 1.5}, "observed_tick must be an integer"),
            ({**base, "observed_tick": True}, "observed_tick must be an integer"),
            ({**base, "order_budget": 0}, "order_budget"),
            ({**base, "order_budget": 5041}, "order_budget"),
            ({**base, "tree_budget": -1}, "tree_budget"),
            ({**base, "tree_budget": 4097}, "tree_budget"),
        ]
        for index, (body, pattern) in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValidationError, pattern):
                    self.service.replan("m1", body, f"bad-{index}")
        with self.assertRaisesRegex(ValidationError, "must be an object"):
            self.service.replan("m1", [1, 2], "bad-body")
        # The boundary values are accepted.
        self.service.replan(
            "m1", {"observed_tick": 100000, "positions": {"a": [4, 0]}}, "edge-tick"
        )

    def test_positions_are_validated(self):
        self.create_solved()
        base = {"observed_tick": 1, "positions": {"a": [1, 0], "b": [4, 1]}}
        cases = [
            ({**base, "positions": {"a": [1, 0]}}, "missing agents: b"),
            ({**base, "positions": {"a": [1, 0], "b": [4, 1], "z": [0, 0]}}, "unknown agents: z"),
            ({**base, "positions": {"a": [2, 1], "b": [4, 1]}}, "obstacle cell"),
            ({**base, "positions": {"a": [5, 0], "b": [4, 1]}}, "between 0 and 4"),
            ({**base, "positions": {"a": [1], "b": [4, 1]}}, r"\[x, y\] pair"),
            ({**base, "positions": {"a": [1, 0], "b": [1, 0]}}, "distinct cell"),
            ({**base, "positions": [[1, 0], [4, 1]]}, "keyed by agent id"),
        ]
        for index, (body, pattern) in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValidationError, pattern):
                    self.service.replan("m1", body, f"bad-{index}")

    def test_arrived_agent_occupies_only_the_observed_tick(self):
        self.create_solved()
        result = self.service.replan(
            "m1", {"observed_tick": 4, "positions": {"a": [4, 0], "b": [2, 2]}}, "r1"
        )
        self.assertEqual("solved", result["status"])
        paths = {entry["agent_id"]: entry["path"] for entry in result["agents"]}
        self.assertEqual([[4, 0]], paths["a"])
        arrivals = {entry["agent_id"]: entry["arrival"] for entry in result["agents"]}
        self.assertEqual(4, arrivals["a"])
        self.assertEqual(["b"], result["priority_order"])
        # b walks straight home along the lower lane and is never blocked by a.
        self.assertEqual([[2, 2], [1, 2], [0, 2], [0, 1], [0, 0]], paths["b"])

    def test_all_agents_arrived_is_a_trivial_solution(self):
        self.create_solved()
        result = self.service.replan(
            "m1", {"observed_tick": 8, "positions": {"a": [4, 0], "b": [0, 0]}}, "r1"
        )
        self.assertEqual("solved", result["status"])
        self.assertEqual(8, result["makespan"])
        self.assertEqual([], result["priority_order"])
        for entry in result["agents"]:
            self.assertEqual(8, entry["arrival"])
            self.assertEqual(1, len(entry["path"]))

    def test_closure_must_not_start_before_the_observed_tick(self):
        self.create_solved([{"id": "a", "start": [0, 0], "goal": [4, 0]}])
        with self.assertRaisesRegex(ValidationError, "must not be earlier than observed_tick"):
            self.service.replan(
                "m1",
                {
                    "observed_tick": 3,
                    "positions": {"a": [3, 0]},
                    "closures": [{"cells": [[4, 0]], "start_tick": 2, "end_tick": 9}],
                },
                "r1",
            )

    def test_closure_forces_a_wait_in_absolute_ticks(self):
        self.service.create_map({"id": "lane", "width": 5, "height": 1, "grid": ["....."]}, "map-lane")
        self.create([{"id": "a", "start": [0, 0], "goal": [4, 0]}], "m1", "lane")
        self.service.plan("m1", {}, "p1")
        result = self.service.replan(
            "m1",
            {
                "observed_tick": 1,
                "positions": {"a": [1, 0]},
                "closures": [{"cells": [[2, 0]], "start_tick": 2, "end_tick": 2}],
            },
            "r1",
        )
        self.assertEqual("solved", result["status"])
        self.assertEqual(
            [[1, 0], [1, 0], [2, 0], [3, 0], [4, 0]], result["agents"][0]["path"]
        )
        self.assertEqual(5, result["agents"][0]["arrival"])
        self.assertEqual(5, result["makespan"])

    def test_closure_covering_a_snapshot_position_is_a_closure_diagnostic(self):
        self.create_solved()
        result = self.service.replan(
            "m1",
            {
                "observed_tick": 2,
                "positions": {"a": [2, 0], "b": [4, 2]},
                "closures": [{"cells": [[2, 0]], "start_tick": 2, "end_tick": 6}],
            },
            "r1",
        )
        self.assertEqual("unsolvable", result["status"])
        self.assertEqual(0, result["orders_tried"])
        self.assertEqual(2, result["replan_from"])
        conflict = result["conflicts"][0]
        self.assertEqual("closure", conflict["type"])
        self.assertEqual("a", conflict["agent"])
        self.assertEqual(2, conflict["timestep"])
        self.assertEqual([2, 0], conflict["position"])

    def test_closure_covering_an_arrived_agent_is_a_closure_diagnostic(self):
        self.create_solved()
        result = self.service.replan(
            "m1",
            {
                "observed_tick": 4,
                "positions": {"a": [4, 0], "b": [2, 2]},
                "closures": [{"cells": [[4, 0]], "start_tick": 4, "end_tick": 4}],
            },
            "r1",
        )
        self.assertEqual("unsolvable", result["status"])
        conflict = result["conflicts"][0]
        self.assertEqual("closure", conflict["type"])
        self.assertEqual("a", conflict["agent"])
        self.assertEqual(4, conflict["timestep"])
        self.assertEqual([4, 0], conflict["position"])

    def test_unsolvable_replan_reports_absolute_timesteps(self):
        self.service.create_map({"id": "lane", "width": 5, "height": 1, "grid": ["....."]}, "map-lane")
        self.create([{"id": "a", "start": [0, 0], "goal": [4, 0]}], "m1", "lane")
        self.service.plan("m1", {}, "p1")
        result = self.service.replan(
            "m1",
            {
                "observed_tick": 3,
                "positions": {"a": [3, 0]},
                "closures": [{"cells": [[4, 0]], "start_tick": 3, "end_tick": 100000}],
            },
            "r1",
        )
        self.assertEqual("unsolvable", result["status"])
        self.assertIsNone(result["makespan"])
        self.assertEqual(3, result["replan_from"])
        self.assertTrue(result["conflicts"])
        for conflict in result["conflicts"]:
            self.assertEqual("closure", conflict["type"])
            self.assertGreaterEqual(conflict["timestep"], 3)

    def test_replan_does_not_touch_the_stored_plan_or_schedule(self):
        plan = self.create_solved()
        before = self.service.get_mission("m1")
        schedule_before = self.service.schedule("m1")
        self.service.replan(
            "m1", {"observed_tick": 2, "positions": {"a": [2, 0], "b": [4, 2]}}, "r1"
        )
        after = self.service.get_mission("m1")
        self.assertEqual(before, after)
        self.assertEqual(plan, after["plan"])
        self.assertEqual("solved", after["status"])
        self.assertEqual(schedule_before, self.service.schedule("m1"))

    def test_replan_is_idempotent_and_survives_a_restart(self):
        path = str(Path(self.directory.name) / "restart.db")
        first = SwarmGrid(path)
        first.create_map(corridor_map(), "map-1")
        first.create_mission(
            {
                "id": "m1",
                "map_id": "corridor",
                "agents": [
                    {"id": "a", "start": [0, 0], "goal": [4, 0]},
                    {"id": "b", "start": [4, 0], "goal": [0, 0]},
                ],
            },
            "k1",
        )
        first.plan("m1", {}, "k2")
        body = {"observed_tick": 2, "positions": {"a": [2, 0], "b": [4, 2]}}
        result = first.replan("m1", body, "r1")
        self.assertEqual(result, first.replan("m1", body, "r1"))
        second = SwarmGrid(path)
        # The memoized key replays the stored bytes, and a fresh key recomputes
        # the same deterministic result.
        self.assertEqual(result, second.replan("m1", body, "r1"))
        self.assertEqual(result, second.replan("m1", body, "r2"))


if __name__ == "__main__":
    unittest.main()
