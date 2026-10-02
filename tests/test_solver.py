import unittest

from swarmgrid.model import Closure, Map, Mission
from swarmgrid.solver import (
    ClosureTable,
    detect_conflicts,
    distance_map,
    makespan,
    plan_mission,
    position_at,
    space_time_astar,
)

OPEN_2X2 = {"id": "square", "width": 2, "height": 2, "grid": ["..", ".."]}


def closure(cells, start, end):
    return Closure(tuple(tuple(cell) for cell in cells), start, end)


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


class ClosureTableTests(unittest.TestCase):
    def test_interval_is_inclusive_on_both_ends(self):
        table = ClosureTable([closure([[1, 0]], 2, 4)])
        self.assertFalse(table.blocks((1, 0), 1))
        self.assertTrue(table.blocks((1, 0), 2))
        self.assertTrue(table.blocks((1, 0), 3))
        self.assertTrue(table.blocks((1, 0), 4))
        self.assertFalse(table.blocks((1, 0), 5))
        self.assertFalse(table.blocks((0, 0), 3))

    def test_overlapping_closures_combine(self):
        table = ClosureTable([closure([[1, 0]], 0, 2), closure([[1, 0], [2, 0]], 2, 5)])
        self.assertTrue(table.blocks((1, 0), 0))
        self.assertTrue(table.blocks((1, 0), 5))
        self.assertFalse(table.blocks((2, 0), 1))
        self.assertTrue(table.blocks((2, 0), 3))


class SingleAgentClosureTests(unittest.TestCase):
    def test_agent_waits_until_a_temporary_gate_opens(self):
        map_record, mission = build(
            {"id": "lane", "width": 5, "height": 1, "grid": ["....."]},
            [{"id": "a", "start": [0, 0], "goal": [4, 0]}],
        )
        agent = mission.agents[0]
        table = ClosureTable([closure([[2, 0]], 2, 2)])
        path, conflict = space_time_astar(
            map_record, agent, [], distance_map(map_record, agent.goal), closures=table
        )
        self.assertIsNone(conflict)
        # The agent waits one tick before entering the gate cell.
        self.assertEqual(((0, 0), (1, 0), (1, 0), (2, 0), (3, 0), (4, 0)), path)
        for timestep, cell in enumerate(path):
            self.assertFalse(table.blocks(cell, timestep))

    def test_closure_after_arrival_does_not_change_the_plan(self):
        map_record, mission = build(
            {"id": "lane", "width": 5, "height": 1, "grid": ["....."]},
            [{"id": "a", "start": [0, 0], "goal": [4, 0]}],
        )
        result = plan_mission(map_record, mission, closures=[closure([[2, 0]], 50, 60)])
        self.assertEqual("solved", result.status)
        self.assertEqual(4, result.solution_makespan)

    def test_a_sealed_corridor_is_reported_as_a_closure_conflict(self):
        map_record, mission = build(
            {"id": "lane", "width": 5, "height": 1, "grid": ["....."]},
            [{"id": "a", "start": [0, 0], "goal": [4, 0]}],
        )
        result = plan_mission(map_record, mission, closures=[closure([[2, 0]], 0, 40)])
        self.assertEqual("unsolvable", result.status)
        self.assertEqual(1, len(result.conflicts))
        blocked = result.conflicts[0]
        self.assertEqual("closure", blocked.type)
        self.assertEqual("a", blocked.agent)
        self.assertEqual((2, 0), blocked.position)
        self.assertIsNone(blocked.other)
        self.assertIsNone(blocked.other_position)
        self.assertEqual(2, blocked.timestep)

    def test_start_covered_by_a_tick_zero_closure_is_diagnosed_up_front(self):
        map_record, mission = build(
            {"id": "lane", "width": 5, "height": 1, "grid": ["....."]},
            [{"id": "a", "start": [0, 0], "goal": [4, 0]}],
        )
        result = plan_mission(map_record, mission, closures=[closure([[0, 0]], 0, 3)])
        self.assertEqual("unsolvable", result.status)
        self.assertEqual(0, result.orders_tried)
        blocked = result.conflicts[0]
        self.assertEqual(("closure", 0, "a", (0, 0)), (blocked.type, blocked.timestep, blocked.agent, blocked.position))

    def test_start_closure_that_opens_after_zero_does_not_seal(self):
        map_record, mission = build(
            {"id": "lane", "width": 5, "height": 1, "grid": ["....."]},
            [{"id": "a", "start": [0, 0], "goal": [4, 0]}],
        )
        result = plan_mission(map_record, mission, closures=[closure([[0, 0]], 1, 2)])
        self.assertEqual("solved", result.status)
        path = result.paths[0]
        self.assertNotEqual(path[1], (0, 0))


class JointClosureTests(unittest.TestCase):
    def _crossing_mission(self):
        return build(
            {"id": "c", "width": 5, "height": 3, "grid": [".....", "..##.", "....."]},
            [
                {"id": "a", "start": [0, 0], "goal": [4, 0]},
                {"id": "b", "start": [4, 0], "goal": [0, 0]},
            ],
        )

    def test_every_published_plan_avoids_the_closed_cells(self):
        map_record, mission = self._crossing_mission()
        result = plan_mission(
            map_record, mission, closures=[closure([[2, 0], [3, 0]], 0, 2)]
        )
        self.assertEqual("solved", result.status)
        agents = tuple(sorted(mission.agents, key=lambda agent: agent.id))
        self.assertEqual([], list(detect_conflicts(agents, list(result.paths))))
        closed = {(2, 0), (3, 0)}
        for path in result.paths:
            for timestep, cell in enumerate(path):
                self.assertFalse(timestep <= 2 and cell in closed)

    def test_multiple_sealed_starts_are_reported_sorted_by_agent_id(self):
        map_record, mission = self._crossing_mission()
        result = plan_mission(
            map_record, mission, closures=[closure([[0, 0], [4, 0]], 0, 0)]
        )
        self.assertEqual("unsolvable", result.status)
        self.assertEqual(0, result.orders_tried)
        self.assertEqual(
            [("closure", 0, "a", (0, 0)), ("closure", 0, "b", (4, 0))],
            [
                (conflict.type, conflict.timestep, conflict.agent, conflict.position)
                for conflict in result.conflicts
            ],
        )

    def test_a_timed_gate_is_handled_by_the_constraint_tree(self):
        map_record, mission = build(
            {"id": "open", "width": 4, "height": 2, "grid": ["....", "...."]},
            [
                {"id": "a", "start": [0, 0], "goal": [3, 0]},
                {"id": "b", "start": [3, 0], "goal": [0, 0]},
            ],
        )
        result = plan_mission(
            map_record,
            mission,
            tree_budget=4096,
            closures=[closure([[0, 1], [1, 1], [2, 1], [3, 1]], 0, 3)],
        )
        self.assertEqual("solved", result.status)
        agents = tuple(sorted(mission.agents, key=lambda agent: agent.id))
        self.assertEqual([], list(detect_conflicts(agents, list(result.paths))))
        for path in result.paths:
            for timestep, cell in enumerate(path):
                self.assertFalse(timestep <= 3 and cell[1] == 1)

    def test_closures_keep_the_plan_deterministic(self):
        map_record, mission = self._crossing_mission()
        payload = [closure([[2, 0]], 1, 2)]
        first = plan_mission(map_record, mission, closures=payload)
        second = plan_mission(map_record, mission, closures=payload)
        self.assertEqual(first, second)


def _agent(agent_id, start, goal):
    from swarmgrid.model import Agent

    return Agent(agent_id, start, goal)


if __name__ == "__main__":
    unittest.main()
