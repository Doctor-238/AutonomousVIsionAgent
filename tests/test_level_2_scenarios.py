import unittest
import asyncio
import os
import sys

# Add repo root to python path to ensure imports work correctly
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from menlo_runner.programs.project.ko.level_2_starter_ko import (
    AgentMemory,
    AgentDecision,
    Observation,
    ScannedDetection,
    _navigation_arrived,
    _pad_candidates,
    _select_target_detection,
    choose_fast_decision,
    parse_task_instructions,
    update_memory,
    validate_decision,
)

class MockRobotState:
    def __init__(self):
        self.held_entity_ids = []
        self.pose = type('Pose', (), {"position": [0, 0, 0], "yaw_deg": 0.0})()
        
class MockStatus:
    def __init__(self):
        self.robot = MockRobotState()


def detection(
    color,
    *,
    area=3000,
    angle=0.0,
    centroid=(160, 180),
    bbox=(130, 150, 60, 60),
):
    return ScannedDetection(
        color=color,
        angle_deg=angle,
        blob_area=area,
        centroid=centroid,
        bbox=bbox,
        head_yaw=0.0,
        head_pitch=0.0,
    )


def observation(*detections):
    return Observation(robot_status=MockStatus(), detections=list(detections))


class Level2ScenarioTest(unittest.TestCase):
    def test_task_parsing_standard(self):
        task = "Find and sort the six cubes in the warehouse into their matching destination pads."
        limit, priorities = asyncio.run(parse_task_instructions(task, ""))
        self.assertIsNone(limit)
        self.assertEqual(priorities, [])

    def test_task_parsing_limit(self):
        task = "6개 중 4개만 목적지 패드로 분류하세요."
        limit, priorities = asyncio.run(parse_task_instructions(task, ""))
        self.assertEqual(limit, 4)

    def test_task_parsing_priority(self):
        task = "노란색과 파란색 큐브를 가장 먼저 처리하고, 나머지를 분류하세요."
        limit, priorities = asyncio.run(parse_task_instructions(task, ""))
        # Check that yellow and blue are extracted in priorities
        priority_lower = [p.lower() for p in priorities]
        self.assertIn("yellow", priority_lower)
        self.assertIn("blue", priority_lower)

    def test_decision_validation_pick_override(self):
        memory = AgentMemory(held_color="red")
        decision = AgentDecision(next_action="pick_cube", target_color="blue")
        validated = validate_decision(decision, memory)
        self.assertEqual(validated.next_action, "search_pad")
        self.assertEqual(validated.target_color, "red")

    def test_decision_validation_place_override(self):
        memory = AgentMemory(held_color=None)
        decision = AgentDecision(next_action="place_cube")
        validated = validate_decision(decision, memory)
        self.assertEqual(validated.next_action, "search_cube")

    def test_decision_validation_priority_ignore_completed(self):
        memory = AgentMemory(completed_colors=["red"])
        decision = AgentDecision(next_action="navigate_to_cube", target_color="red")
        validated = validate_decision(decision, memory)
        self.assertIsNone(validated.target_color)

    def test_fast_decision_navigates_before_pick(self):
        memory = AgentMemory()
        decision = choose_fast_decision(observation(detection("red")), memory)
        self.assertEqual(decision.next_action, "navigate_to_cube")
        self.assertEqual(decision.target_color, "red")

    def test_fast_decision_picks_only_after_cube_ready(self):
        memory = AgentMemory(active_color="red", cube_ready=True)
        decision = choose_fast_decision(observation(detection("red")), memory)
        self.assertEqual(decision.next_action, "pick_cube")
        self.assertEqual(decision.target_color, "red")

    def test_validation_blocks_pick_without_cube_navigation(self):
        memory = AgentMemory(active_color="red", cube_ready=False)
        decision = AgentDecision(next_action="pick_cube", target_color="red")
        validated = validate_decision(decision, memory)
        self.assertEqual(validated.next_action, "navigate_to_cube")
        self.assertEqual(validated.target_color, "red")

    def test_validation_blocks_place_without_pad_navigation(self):
        memory = AgentMemory(held_color="red", pad_ready=False)
        decision = AgentDecision(next_action="place_cube", target_color="red")
        validated = validate_decision(decision, memory)
        self.assertEqual(validated.next_action, "navigate_to_pad")
        self.assertEqual(validated.target_color, "red")

    def test_fast_decision_ignores_carried_cube_when_searching_pad(self):
        memory = AgentMemory(held_color="red")
        carried_cube_blob = detection(
            "red",
            area=12000,
            centroid=(160, 320),
            bbox=(115, 250, 90, 90),
        )
        decision = choose_fast_decision(observation(carried_cube_blob), memory)
        self.assertEqual(decision.next_action, "search_pad")
        self.assertEqual(decision.target_color, "red")

    def test_fast_decision_recovers_after_failed_navigation(self):
        memory = AgentMemory(held_color="blue")
        last_result = {
            "decision": {"next_action": "navigate_to_pad", "target_color": "blue"},
            "action_result": {"action": "navigate_to_pad", "reached": False},
        }
        decision = choose_fast_decision(observation(detection("blue")), memory, last_result)
        self.assertEqual(decision.next_action, "recover")
        self.assertEqual(decision.target_color, "blue")

    def test_pad_candidates_filter_large_held_blob(self):
        large_held_blob = detection("blue", area=43000, angle=13.0)
        usable_pad_blob = detection("blue", area=26000, angle=-5.0)
        candidates = _pad_candidates([large_held_blob, usable_pad_blob], "blue")
        self.assertEqual(candidates, [usable_pad_blob])

    def test_pad_selector_prefers_marker_over_floor_band(self):
        memory = AgentMemory(held_color="blue")
        floor_band = detection(
            "blue",
            area=30000,
            angle=0.0,
            centroid=(500, 709),
            bbox=(0, 699, 1000, 21),
        )
        marker = detection(
            "blue",
            area=6000,
            angle=6.0,
            centroid=(980, 70),
            bbox=(930, 30, 100, 80),
        )
        selected = _select_target_detection([floor_band, marker], "blue", "pad", memory)
        self.assertEqual(selected, marker)

    def test_pad_selector_uses_tracking_continuity(self):
        memory = AgentMemory(
            held_color="blue",
            nav_track_kind="pad",
            nav_track_color="blue",
            nav_track_angle=22.0,
        )
        larger_far_candidate = detection(
            "blue",
            area=26000,
            angle=-14.0,
            centroid=(390, 120),
            bbox=(300, 70, 180, 100),
        )
        tracked_candidate = detection(
            "blue",
            area=8000,
            angle=20.0,
            centroid=(950, 120),
            bbox=(900, 80, 90, 80),
        )
        selected = _select_target_detection(
            [larger_far_candidate, tracked_candidate],
            "blue",
            "pad",
            memory,
        )
        self.assertEqual(selected, tracked_candidate)

    def test_cube_selector_avoids_wide_floor_band(self):
        memory = AgentMemory()
        floor_band = detection(
            "red",
            area=50000,
            angle=0.0,
            centroid=(400, 520),
            bbox=(0, 500, 800, 40),
        )
        cube_blob = detection(
            "red",
            area=12000,
            angle=8.0,
            centroid=(620, 310),
            bbox=(580, 270, 80, 80),
        )
        selected = _select_target_detection([floor_band, cube_blob], "red", "cube", memory)
        self.assertEqual(selected, cube_blob)

    def test_update_memory_counts_only_real_place_success(self):
        memory = AgentMemory(held_color="red", pad_ready=True)
        update_memory(
            memory,
            observation(),
            AgentDecision(next_action="place_cube", target_color="red"),
            {"action_result": {"held": False, "status": "success"}},
        )
        self.assertEqual(memory.delivered_count, 1)
        self.assertIsNone(memory.held_color)

    def test_update_memory_does_not_count_place_when_not_holding(self):
        memory = AgentMemory(held_color=None, pad_ready=True)
        update_memory(
            memory,
            observation(),
            AgentDecision(next_action="place_cube", target_color="red"),
            {"action_result": {"held": False, "status": "success"}},
        )
        self.assertEqual(memory.delivered_count, 0)
        self.assertIsNone(memory.held_color)

    def test_cube_arrival_requires_centered_target(self):
        arrived = _navigation_arrived(
            target_kind="cube",
            area=21250,
            angle_deg=24.3,
            moved_toward_target=False,
            pad_direction_confirmed=False,
            pad_forward_steps=0,
            step=1,
        )
        self.assertFalse(arrived)

    def test_cube_arrival_allows_close_centered_target(self):
        arrived = _navigation_arrived(
            target_kind="cube",
            area=34599,
            angle_deg=-5.0,
            moved_toward_target=False,
            pad_direction_confirmed=False,
            pad_forward_steps=0,
            step=1,
        )
        self.assertTrue(arrived)

    def test_cube_arrival_allows_nearly_centered_close_target(self):
        arrived = _navigation_arrived(
            target_kind="cube",
            area=45301,
            angle_deg=-9.4,
            moved_toward_target=False,
            pad_direction_confirmed=False,
            pad_forward_steps=0,
            step=5,
        )
        self.assertTrue(arrived)

    def test_pad_arrival_requires_centered_target(self):
        arrived = _navigation_arrived(
            target_kind="pad",
            area=54438,
            angle_deg=-25.1,
            moved_toward_target=True,
            pad_direction_confirmed=False,
            pad_forward_steps=4,
            step=10,
        )
        self.assertFalse(arrived)

    def test_pad_arrival_allows_centered_after_forward_motion(self):
        arrived = _navigation_arrived(
            target_kind="pad",
            area=26047,
            angle_deg=-5.3,
            moved_toward_target=True,
            pad_direction_confirmed=False,
            pad_forward_steps=4,
            step=10,
        )
        self.assertTrue(arrived)

    def test_pad_arrival_allows_small_centered_sign_after_forward_motion(self):
        arrived = _navigation_arrived(
            target_kind="pad",
            area=5370,
            angle_deg=-5.8,
            moved_toward_target=True,
            pad_direction_confirmed=False,
            pad_forward_steps=3,
            step=7,
        )
        self.assertTrue(arrived)


if __name__ == "__main__":
    unittest.main()
