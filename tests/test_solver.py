import unittest

from swarmgrid.model import Map, Mission, parse_closures
from swarmgrid.solver import (
    detect_conflicts,
    distance_map,
    makespan,
    plan_mission,
    position_at,
    space_time_astar,
)

OPEN_2X2 = {"id": "square", "width": 2, "height": 2, "grid": ["..", ".."]}


def build(map_payload, agents, mission_id="m"):
    map_record = Map.parse(map_payload)
    mission = Mission.parse({"id": mission_id, "map_id": map_payload["id"], "agents": agents}, map_record)
    return map_record, mission


class SingleAgentTests(unittest.TestCase):
    def test_shortest_path_avoids_obstacles(self):
        map_record, mission = build(
            {"id": "w", "width": 5, "height": 2, "grid": ["..#..", "....."]},
            [{"id": "a", "start": [0, 0], "goal": [4, 0]}],
        )
        agent = mission.agents[0]
        path, conflict = space_time_astar(map_record, agent, [], distance_map(map_record, agent.goal))
        self.assertIsNone(conflict)
        self.assertEqual((0, 0), path[0])
        self.assertEqual((4, 0), path[-1])
        self.assertEqual(6, makespan([path]))
        self.assertNotIn((2, 0), path)

    def test_waiting_is_used_when_a_cell_is_blocked_for_one_tick(self):
        map_record, mission = build(
            OPEN_2X2,
            [{"id": "a", "start": [0, 0], "goal": [1, 0]}],
        )
        agent = mission.agents[0]
        blocked = frozenset({((1, 0), 1)})
        path, conflict = space_time_astar(
            map_record, agent, [], distance_map(map_record, agent.goal), blocked
        )
        self.assertIsNone(conflict)
        self.assertEqual(((0, 0), (0, 0), (1, 0)), path)

    def test_unreachable_goal_reports_a_conflict(self):
        map_record, mission = build(
            {"id": "w", "width": 5, "height": 3, "grid": [".#...", "##...", "....."]},
            [{"id": "a", "start": [0, 0], "goal": [0, 2]}],
        )
        agent = mission.agents[0]
        path, conflict = space_time_astar(map_record, agent, [], distance_map(map_record, agent.goal))
        self.assertIsNone(path)
        self.assertEqual("unreachable_goal", conflict.type)


class OccupancyTests(unittest.TestCase):
    def test_a_finished_agent_no_longer_occupies_its_goal(self):
        path = ((0, 0), (1, 0), (0, 0))
        self.assertEqual((1, 0), position_at(path, 1))
        self.assertIsNone(position_at(path, 3))

    def test_following_is_legal(self):
        lead = ((0, 0), (1, 0), (2, 0))
        trail = ((1, 0), (1, 0), (1, 0), (1, 0))
        agents = (
            _agent("lead", (0, 0), (2, 0)),
            _agent("trail", (1, 0), (1, 0)),
        )
        # trail occupies (1, 0) from t=0; lead passes through it at t=1.
        # Sharing a cell is never allowed, so this plan must be reported.
        conflicts = detect_conflicts(agents, [lead, trail])
        self.assertTrue(any(conflict.type == "vertex" for conflict in conflicts))

        # When trail leaves on the same tick lead arrives, the two never share
        # the cell and the plan is clean.
        moving = ((1, 0), (1, 1), (1, 2), (2, 2))
        self.assertEqual([], detect_conflicts(agents, [lead, moving]))

    def test_a_swap_is_reported(self):
        first = ((0, 0), (1, 0))
        second = ((1, 0), (0, 0))
        agents = (_agent("a", (0, 0), (1, 0)), _agent("b", (1, 0), (0, 0)))
        conflicts = detect_conflicts(agents, [first, second])
        self.assertEqual(["edge"], [conflict.type for conflict in conflicts])
        self.assertEqual(0, conflicts[0].timestep)

    def test_sharing_a_cell_is_reported(self):
        first = ((0, 0), (1, 0), (1, 0))
        second = ((1, 0), (1, 0), (0, 0))
        agents = (_agent("a", (0, 0), (1, 0)), _agent("b", (1, 0), (0, 0)))
        conflicts = detect_conflicts(agents, [first, second])
        self.assertTrue(any(conflict.type == "vertex" for conflict in conflicts))

    def test_two_agents_settling_on_one_cell_is_a_conflict(self):
        first = ((0, 0), (1, 0))
        second = ((1, 1), (1, 0))
        agents = (_agent("a", (0, 0), (1, 0)), _agent("b", (1, 1), (1, 0)))
        conflicts = detect_conflicts(agents, [first, second])
        self.assertTrue(any(conflict.type == "vertex" for conflict in conflicts))


class JointPlanningTests(unittest.TestCase):
    def test_opposite_agents_in_a_corridor_are_routed_around_each_other(self):
        from swarmgrid.solver import plan_mission

        map_record, mission = build(
            {"id": "c", "width": 5, "height": 3, "grid": [".....", "..##.", "....."]},
            [
                {"id": "a", "start": [0, 0], "goal": [4, 0]},
                {"id": "b", "start": [4, 0], "goal": [0, 0]},
            ],
        )
        result = plan_mission(map_record, mission)
        self.assertEqual("solved", result.status)
        self.assertEqual([], list(result.conflicts))
        self.assertEqual(8, result.solution_makespan)
        self.assertEqual([], detect_conflicts(
            tuple(sorted(mission.agents, key=lambda agent: agent.id)), list(result.paths)
        ))

    def test_the_constraint_tree_solves_what_a_single_order_cannot(self):
        from swarmgrid.solver import _constraint_tree, plan_mission

        map_record, mission = build(
            OPEN_2X2,
            [
                {"id": "a0", "start": [0, 0], "goal": [1, 1]},
                {"id": "a1", "start": [1, 0], "goal": [0, 0]},
            ],
        )
        agents = tuple(sorted(mission.agents, key=lambda agent: agent.id))
        distances = {agent.goal: distance_map(map_record, agent.goal) for agent in agents}
        paths, conflicts = _constraint_tree(map_record, agents, distances, 64)
        self.assertEqual([], conflicts)
        self.assertEqual([], detect_conflicts(agents, paths))
        self.assertEqual(3, makespan(paths))
        self.assertEqual(((0, 0), (1, 0), (1, 1)), tuple(paths[0]))
        self.assertEqual(((1, 0), (1, 1), (0, 1), (0, 0)), tuple(paths[1]))

        priority = plan_mission(map_record, mission)
        self.assertEqual("solved", priority.status)
        self.assertEqual("priority", priority.method)
        self.assertEqual(tuple(paths), priority.paths)

    def test_the_constraint_tree_stops_at_its_node_budget(self):
        from swarmgrid.solver import _constraint_tree

        map_record, mission = build(
            {"id": "lane", "width": 4, "height": 1, "grid": ["...."]},
            [
                {"id": "a", "start": [0, 0], "goal": [3, 0]},
                {"id": "b", "start": [3, 0], "goal": [0, 0]},
            ],
        )
        agents = tuple(sorted(mission.agents, key=lambda agent: agent.id))
        distances = {agent.goal: distance_map(map_record, agent.goal) for agent in agents}
        paths, conflicts = _constraint_tree(map_record, agents, distances, 0)
        self.assertTrue(conflicts)
        self.assertEqual([], list(paths))

    def test_swapping_agents_in_a_single_lane_stay_unsolvable(self):
        from swarmgrid.solver import plan_mission

        map_record, mission = build(
            {"id": "lane", "width": 4, "height": 1, "grid": ["...."]},
            [
                {"id": "a", "start": [0, 0], "goal": [3, 0]},
                {"id": "b", "start": [3, 0], "goal": [0, 0]},
            ],
        )
        result = plan_mission(map_record, mission, tree_budget=256)
        self.assertEqual("unsolvable", result.status)
        self.assertTrue(result.conflicts)
        self.assertEqual([], list(result.paths))

    def test_a_free_map_needs_no_replanning(self):
        from swarmgrid.solver import plan_mission

        map_record, mission = build(
            {"id": "open", "width": 6, "height": 4, "grid": ["......"] * 4},
            [
                {"id": "a", "start": [0, 0], "goal": [5, 0]},
                {"id": "b", "start": [5, 3], "goal": [0, 3]},
            ],
        )
        result = plan_mission(map_record, mission)
        self.assertEqual("solved", result.status)
        self.assertEqual("priority", result.method)
        self.assertEqual(5, result.solution_makespan)


LANE = {"id": "lane", "width": 5, "height": 1, "grid": ["....."]}


class ClosureTests(unittest.TestCase):
    def test_single_agent_waits_out_a_one_tick_closure(self):
        map_record, mission = build(
            LANE, [{"id": "a", "start": [0, 0], "goal": [4, 0]}]
        )
        closures = parse_closures([{"cells": [[1, 0]], "start_tick": 1, "end_tick": 1}], map_record)
        result = plan_mission(map_record, mission, closures=closures)
        self.assertEqual("solved", result.status)
        self.assertEqual(
            ((0, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0)), result.paths[0]
        )

    def test_the_cell_is_free_again_after_the_interval(self):
        map_record, mission = build(
            LANE, [{"id": "a", "start": [0, 0], "goal": [4, 0]}]
        )
        # Closed only at t=1: one wait. Closed at t=2: arrive one tick later by
        # waiting, never occupying the cell on the closed tick.
        closures = parse_closures([{"cells": [[2, 0]], "start_tick": 2, "end_tick": 2}], map_record)
        result = plan_mission(map_record, mission, closures=closures)
        self.assertEqual("solved", result.status)
        path = result.paths[0]
        self.assertNotEqual((2, 0), path[2])
        self.assertEqual(5, makespan([path]))

    def test_a_permanent_closure_is_an_unsolvable_closure_conflict(self):
        map_record, mission = build(
            LANE, [{"id": "a", "start": [0, 0], "goal": [4, 0]}]
        )
        closures = parse_closures(
            [{"cells": [[1, 0], [2, 0], [3, 0]], "start_tick": 0, "end_tick": 100000}],
            map_record,
        )
        result = plan_mission(map_record, mission, closures=closures)
        self.assertEqual("unsolvable", result.status)
        self.assertEqual((), result.paths)
        conflict = result.conflicts[0]
        self.assertEqual("closure", conflict.type)
        self.assertIsNone(conflict.other)
        self.assertIsNone(conflict.other_position)
        self.assertEqual("a", conflict.agent)
        self.assertEqual(1, conflict.timestep)
        self.assertEqual((1, 0), conflict.position)

    def test_a_start_covered_at_tick_zero_is_diagnosed_per_agent_sorted(self):
        map_record, mission = build(
            LANE,
            [
                {"id": "b", "start": [4, 0], "goal": [0, 0]},
                {"id": "a", "start": [0, 0], "goal": [4, 0]},
            ],
        )
        closures = parse_closures(
            [{"cells": [[0, 0], [4, 0]], "start_tick": 0, "end_tick": 3}], map_record
        )
        result = plan_mission(map_record, mission, closures=closures)
        self.assertEqual("unsolvable", result.status)
        self.assertEqual(0, result.orders_tried)
        self.assertEqual(
            [
                ("closure", 0, "a", (0, 0)),
                ("closure", 0, "b", (4, 0)),
            ],
            [(c.type, c.timestep, c.agent, c.position) for c in result.conflicts],
        )
        for conflict in result.conflicts:
            self.assertIsNone(conflict.other)
            self.assertIsNone(conflict.other_position)

    def test_only_the_agent_whose_start_is_closed_is_diagnosed(self):
        map_record, mission = build(
            LANE,
            [
                {"id": "a", "start": [0, 0], "goal": [4, 0]},
                {"id": "b", "start": [4, 0], "goal": [0, 0]},
            ],
        )
        # Only b's start is closed at t=0; a remains free to move.
        closures = parse_closures([{"cells": [[4, 0]], "start_tick": 0, "end_tick": 0}], map_record)
        result = plan_mission(map_record, mission, closures=closures)
        self.assertEqual("unsolvable", result.status)
        self.assertEqual(
            [("closure", 0, "b", (4, 0))],
            [(c.type, c.timestep, c.agent, c.position) for c in result.conflicts],
        )

    def test_a_closure_starting_after_tick_zero_does_not_block_the_start(self):
        map_record, mission = build(
            LANE, [{"id": "a", "start": [0, 0], "goal": [4, 0]}]
        )
        closures = parse_closures([{"cells": [[0, 0]], "start_tick": 1, "end_tick": 1}], map_record)
        result = plan_mission(map_record, mission, closures=closures)
        self.assertEqual("solved", result.status)
        # The agent simply leaves its start at t=0; it never waits on the closed
        # cell at t=1.
        self.assertNotEqual((0, 0), result.paths[0][1])

    def test_a_closure_after_the_agent_finishes_is_irrelevant(self):
        map_record, mission = build(
            LANE, [{"id": "a", "start": [0, 0], "goal": [4, 0]}]
        )
        closures = parse_closures([{"cells": [[4, 0]], "start_tick": 10, "end_tick": 20}], map_record)
        result = plan_mission(map_record, mission, closures=closures)
        self.assertEqual("solved", result.status)
        self.assertEqual(4, result.solution_makespan)

    def test_closures_are_honoured_by_the_constraint_tree(self):
        from swarmgrid.solver import _constraint_tree

        map_record, mission = build(
            OPEN_2X2,
            [
                {"id": "a0", "start": [0, 0], "goal": [1, 1]},
                {"id": "a1", "start": [1, 0], "goal": [0, 0]},
            ],
        )
        agents = tuple(sorted(mission.agents, key=lambda agent: agent.id))
        distances = {agent.goal: distance_map(map_record, agent.goal) for agent in agents}
        closures = parse_closures([{"cells": [[0, 1]], "start_tick": 1, "end_tick": 1}], map_record)
        paths, conflicts = _constraint_tree(map_record, agents, distances, 64, closures=closures)
        self.assertEqual([], conflicts)
        self.assertEqual([], detect_conflicts(agents, paths))
        for path in paths:
            self.assertNotEqual((0, 1), position_at(path, 1))

    def test_closures_keep_planning_deterministic(self):
        map_record, mission = build(
            OPEN_2X2,
            [
                {"id": "a0", "start": [0, 0], "goal": [1, 1]},
                {"id": "a1", "start": [1, 0], "goal": [0, 0]},
            ],
        )
        payload = [{"cells": [[0, 1]], "start_tick": 2, "end_tick": 3}]
        first = plan_mission(map_record, mission, closures=parse_closures(payload, map_record))
        second = plan_mission(map_record, mission, closures=parse_closures(payload, map_record))
        self.assertEqual(first, second)

    def test_closures_route_two_agents_without_collisions(self):
        map_record, mission = build(
            {"id": "c", "width": 5, "height": 3, "grid": [".....", "..##.", "....."]},
            [
                {"id": "a", "start": [0, 0], "goal": [4, 0]},
                {"id": "b", "start": [4, 0], "goal": [0, 0]},
            ],
        )
        closures = parse_closures(
            [{"cells": [[4, 1], [4, 2]], "start_tick": 0, "end_tick": 1}], map_record
        )
        result = plan_mission(map_record, mission, closures=closures)
        self.assertEqual("solved", result.status)
        agents = tuple(sorted(mission.agents, key=lambda agent: agent.id))
        self.assertEqual([], detect_conflicts(agents, list(result.paths)))
        for path in result.paths:
            for timestep, cell in enumerate(path):
                if timestep <= 1:
                    self.assertNotIn(cell, ((4, 1), (4, 2)))


def _agent(agent_id, start, goal):
    from swarmgrid.model import Agent

    return Agent(agent_id, start, goal)


if __name__ == "__main__":
    unittest.main()
