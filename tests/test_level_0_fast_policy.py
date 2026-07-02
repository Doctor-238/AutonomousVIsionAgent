import unittest
from types import SimpleNamespace
from unittest.mock import patch

from menlo_runner.programs.project.ko.level_0_starter_ko import (
    AgentDecision,
    AgentMemory,
    Observation,
    _choose_priority_colors,
    _select_source_target,
    choose_fast_decision,
    update_memory,
)


def pose(x, y):
    return SimpleNamespace(position=(x, y, 0.0))


def entity(x, y, *, visible=True, color=None, attached_to=None):
    state = {"color": color} if color else {}
    return SimpleNamespace(
        pose=pose(x, y),
        visible=visible,
        state=state,
        attached_to=attached_to,
    )


def scene(**entities):
    return SimpleNamespace(entities=entities)


def observation(cubes, held=None):
    return Observation(
        robot_status=None,
        visible_cubes=cubes,
        held_cube=held,
        delivered_cube_ids=[],
        color_to_pad={"red": "pad_B", "green": "pad_C", "blue": "pad_D", "yellow": "pad_E"},
    )


class Level0FastPolicyTest(unittest.TestCase):
    def test_choose_priority_colors_from_source_route_cost(self):
        test_scene = scene(
            pad_A=entity(0.0, 0.0),
            pad_B=entity(1.0, 0.0),
            pad_D=entity(2.0, 0.0),
            pad_C=entity(7.0, 0.0),
            pad_E=entity(8.0, 0.0),
            robot=entity(0.0, 0.0),
        )

        with patch.dict("os.environ", {}, clear=False):
            priority, scores, source = _choose_priority_colors(test_scene)

        self.assertEqual(priority, ["red", "blue"])
        self.assertEqual(source, (0.0, 0.0))
        self.assertLess(scores["red"], scores["green"])
        self.assertLess(scores["blue"], scores["yellow"])

    def test_source_selector_delivers_priority_cube_at_front(self):
        memory = AgentMemory(priority_colors=["red", "blue"], source_position=(0.0, 0.0))
        obs = observation(
            [
                {
                    "entity_id": "cube_pool_0",
                    "color": "blue",
                    "position": (0.2, 0.0, 0.0),
                    "distance_from_robot": 0.4,
                },
                {
                    "entity_id": "cube_pool_1",
                    "color": "yellow",
                    "position": (0.6, 0.0, 0.0),
                    "distance_from_robot": 0.7,
                },
            ]
        )

        target, mode = _select_source_target(obs, memory)

        self.assertEqual(target["entity_id"], "cube_pool_0")
        self.assertEqual(mode, "deliver")

    def test_source_selector_lookahead_can_skip_near_unwanted_cube(self):
        memory = AgentMemory(priority_colors=["red", "blue"], source_position=(0.0, 0.0))
        obs = observation(
            [
                {
                    "entity_id": "cube_pool_0",
                    "color": "yellow",
                    "position": (0.2, 0.0, 0.0),
                    "distance_from_robot": 0.4,
                },
                {
                    "entity_id": "cube_pool_1",
                    "color": "red",
                    "position": (0.9, 0.0, 0.0),
                    "distance_from_robot": 1.0,
                },
            ]
        )

        target, mode = _select_source_target(obs, memory)

        self.assertEqual(target["entity_id"], "cube_pool_1")
        self.assertEqual(mode, "deliver")

    def test_source_selector_discards_front_unwanted_when_priority_is_too_far(self):
        memory = AgentMemory(priority_colors=["red", "blue"], source_position=(0.0, 0.0))
        obs = observation(
            [
                {
                    "entity_id": "cube_pool_0",
                    "color": "green",
                    "position": (0.2, 0.0, 0.0),
                    "distance_from_robot": 0.4,
                },
                {
                    "entity_id": "cube_pool_1",
                    "color": "red",
                    "position": (2.4, 0.0, 0.0),
                    "distance_from_robot": 2.6,
                },
            ]
        )

        target, mode = _select_source_target(obs, memory)

        self.assertEqual(target["entity_id"], "cube_pool_0")
        self.assertEqual(mode, "discard")

    def test_fast_decision_keeps_pick_after_successful_cube_navigation(self):
        memory = AgentMemory(
            priority_colors=["red", "blue"],
            active_cube_id="cube_pool_7",
            active_color="red",
            stage="ready_pick",
        )

        decision = choose_fast_decision(observation([]), memory)

        self.assertEqual(decision.next_action, "pick_cube")
        self.assertEqual(decision.target_entity_id, "cube_pool_7")

    def test_update_memory_counts_delivery_and_discard_separately(self):
        memory = AgentMemory(
            priority_colors=["red", "blue"],
            held_color="red",
            held_entity_id="cube_pool_3",
            active_mode="deliver",
            stage="ready_place",
        )
        obs = observation([], held={"entity_id": "cube_pool_3", "color": "red"})
        verified = {
            "action_result": {"action": "place_cube", "ok": True, "was_holding": True, "mode": "deliver"},
            "held_cube": None,
            "delivered_cube_ids": ["cube_pool_3"],
        }

        update_memory(memory, obs, AgentDecision("place_cube", "red", "pad_B"), verified)

        self.assertEqual(memory.delivered_count, 1)
        self.assertEqual(memory.discarded_count, 0)

        memory.held_color = "green"
        memory.held_entity_id = "cube_pool_4"
        memory.active_mode = "discard"
        obs = observation([], held={"entity_id": "cube_pool_4", "color": "green"})
        verified["action_result"] = {
            "action": "place_cube",
            "ok": True,
            "was_holding": True,
            "mode": "discard",
        }
        verified["delivered_cube_ids"] = ["cube_pool_3", "cube_pool_4"]

        update_memory(memory, obs, AgentDecision("place_cube", "green", "pad_A"), verified)

        self.assertEqual(memory.delivered_count, 1)
        self.assertEqual(memory.discarded_count, 1)


if __name__ == "__main__":
    unittest.main()
