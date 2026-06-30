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
    AgentDecision,
    AgentMemory,
    Observation,
    ScannedDetection,
    _head_pitch_for_target,
    _navigation_arrived,
    _navigation_velocity_command,
    _pad_candidates,
    _select_target_detection,
    _should_peek_down,
    choose_fast_decision,
    get_robot_status,
    move_velocity,
    parse_task_instructions,
    result_summary,
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


class FailingStatusContext:
    async def state(self, key):
        raise TimeoutError(f"{key} unavailable")


class FailingInvokeContext:
    async def invoke(self, skill_name, args, timeout_s=None):
        raise TimeoutError(f"{skill_name} rpc timeout")


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
        limit, priorities, skipped = asyncio.run(parse_task_instructions(task, ""))
        self.assertIsNone(limit)
        self.assertEqual(priorities, [])
        self.assertEqual(skipped, [])

    def test_task_parsing_limit(self):
        task = "6개 중 4개만 목적지 패드에 분류하세요."
        limit, priorities, skipped = asyncio.run(parse_task_instructions(task, ""))
        self.assertEqual(limit, 4)

    def test_task_parsing_priority(self):
        task = "노란색과 파란색 큐브를 가장 먼저 처리하고, 나머지를 분류하세요."
        limit, priorities, skipped = asyncio.run(parse_task_instructions(task, ""))
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
        cube_blob = detection(
            "red",
            area=3600,
            angle=-1.5,
            centroid=(613, 408),
            bbox=(567, 381, 92, 53),
        )
        decision = choose_fast_decision(observation(cube_blob), memory)
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

    def test_pad_candidates_filter_small_carried_cube_blob(self):
        carried_cube_blob = detection(
            "blue",
            area=2366,
            angle=-0.9,
            centroid=(620, 347),
            bbox=(595, 321, 52, 55),
        )
        usable_pad_marker = detection(
            "blue",
            area=9000,
            angle=8.0,
            centroid=(850, 80),
            bbox=(790, 25, 120, 90),
        )
        candidates = _pad_candidates([carried_cube_blob, usable_pad_marker], "blue")
        self.assertEqual(candidates, [usable_pad_marker])

    def test_pad_candidates_require_container_marker_shape(self):
        low_color_square = detection(
            "blue",
            area=8500,
            angle=2.0,
            centroid=(615, 430),
            bbox=(560, 380, 110, 100),
        )
        usable_pad_marker = detection(
            "blue",
            area=7600,
            angle=-4.0,
            centroid=(770, 145),
            bbox=(720, 95, 100, 90),
        )
        candidates = _pad_candidates([low_color_square, usable_pad_marker], "blue")
        self.assertEqual(candidates, [usable_pad_marker])

    def test_fast_decision_uses_known_pad_bearing(self):
        memory = AgentMemory(
            held_color="blue",
            known_pad_bearings={"blue": {"bearing_deg": -35.0, "area": 9000}},
        )
        decision = choose_fast_decision(observation(), memory)
        self.assertEqual(decision.next_action, "navigate_to_pad")
        self.assertEqual(decision.target_color, "blue")

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

    def test_cube_selector_rejects_live_floor_strip_shape(self):
        memory = AgentMemory()
        live_green_strip = detection(
            "green",
            area=32961,
            angle=-3.4,
            centroid=(568, 579),
            bbox=(255, 528, 618, 88),
        )
        blue_cube = detection(
            "blue",
            area=9000,
            angle=2.0,
            centroid=(620, 310),
            bbox=(580, 270, 80, 80),
        )
        self.assertIsNone(_select_target_detection([live_green_strip], "green", "cube", memory))
        selected = _select_target_detection([live_green_strip, blue_cube], None, "cube", memory)
        self.assertEqual(selected, blue_cube)

    def test_cube_selector_rejects_live_overhead_sign_shape(self):
        memory = AgentMemory()
        live_blue_sign = detection(
            "blue",
            area=18699,
            angle=5.6,
            centroid=(760, 57),
            bbox=(685, 0, 203, 138),
        )
        green_cube = detection(
            "green",
            area=9000,
            angle=-2.0,
            centroid=(620, 330),
            bbox=(580, 290, 80, 80),
        )
        self.assertIsNone(_select_target_detection([live_blue_sign], "blue", "cube", memory))
        selected = _select_target_detection([live_blue_sign, green_cube], None, "cube", memory)
        self.assertEqual(selected, green_cube)

    def test_cube_selector_rejects_destination_marker_shape(self):
        memory = AgentMemory()
        blue_destination_marker = detection(
            "blue",
            area=7600,
            angle=3.0,
            centroid=(770, 145),
            bbox=(720, 95, 100, 90),
        )
        real_blue_cube = detection(
            "blue",
            area=3600,
            angle=-1.5,
            centroid=(613, 408),
            bbox=(567, 381, 92, 53),
        )
        self.assertIsNone(_select_target_detection([blue_destination_marker], "blue", "cube", memory))
        selected = _select_target_detection(
            [blue_destination_marker, real_blue_cube],
            "blue",
            "cube",
            memory,
        )
        self.assertEqual(selected, real_blue_cube)

    def test_cube_selector_rejects_live_edge_scene_blob(self):
        memory = AgentMemory()
        edge_blob = detection(
            "blue",
            area=46107,
            angle=-15.5,
            centroid=(309, 517),
            bbox=(0, 356, 525, 310),
        )
        centered_cube = detection(
            "blue",
            area=12000,
            angle=-3.0,
            centroid=(590, 350),
            bbox=(550, 310, 85, 85),
        )
        self.assertIsNone(_select_target_detection([edge_blob], "blue", "cube", memory))
        selected = _select_target_detection([edge_blob, centered_cube], "blue", "cube", memory)
        self.assertEqual(selected, centered_cube)

    def test_cube_selector_rejects_live_wide_low_patch(self):
        memory = AgentMemory()
        low_patch = detection(
            "green",
            area=51547,
            angle=0.0,
            centroid=(640, 617),
            bbox=(329, 541, 567, 137),
        )
        real_cube = detection(
            "green",
            area=9000,
            angle=3.0,
            centroid=(600, 360),
            bbox=(560, 320, 85, 85),
        )
        self.assertIsNone(_select_target_detection([low_patch], "green", "cube", memory))
        selected = _select_target_detection([low_patch, real_cube], "green", "cube", memory)
        self.assertEqual(selected, real_cube)

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

    def test_robot_status_fallback_on_timeout(self):
        memory = AgentMemory()
        status = asyncio.run(get_robot_status(FailingStatusContext(), memory))
        self.assertEqual(status.robot.held_entity_ids, [])
        self.assertEqual(memory.robot_status_failures, 1)

    def test_robot_status_reuses_last_known_status(self):
        memory = AgentMemory(last_robot_status=MockStatus())
        status = asyncio.run(get_robot_status(FailingStatusContext(), memory))
        self.assertIs(status, memory.last_robot_status)
        self.assertEqual(memory.robot_status_failures, 1)

    def test_move_velocity_returns_bounded_failure_on_rpc_timeout(self):
        result = asyncio.run(move_velocity(FailingInvokeContext(), vx=0.2, duration_s=0.1))
        self.assertEqual(result["status"], "failed")
        self.assertIn("TimeoutError", result["error"])
        summary = result_summary(result)
        self.assertEqual(summary["status"], "failed")
        self.assertIn("set_velocity", summary["error"])

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

    def test_cube_arrival_allows_sim_pickup_distance_after_approach(self):
        arrived = _navigation_arrived(
            target_kind="cube",
            area=5706,
            angle_deg=-8.4,
            moved_toward_target=True,
            pad_direction_confirmed=False,
            pad_forward_steps=0,
            step=9,
        )
        self.assertTrue(arrived)

    def test_cube_arrival_rejects_initial_midsize_blob(self):
        arrived = _navigation_arrived(
            target_kind="cube",
            area=3505,
            angle_deg=-1.5,
            moved_toward_target=False,
            pad_direction_confirmed=False,
            pad_forward_steps=0,
            step=1,
        )
        self.assertFalse(arrived)

    def test_cube_servo_is_cautious_before_contact(self):
        vx, vy, wz, duration = _navigation_velocity_command(
            target_kind="cube",
            area=3000,
            angle=2.0,
            arrival_area=10000,
        )
        self.assertGreater(vx, 0.0)
        self.assertLessEqual(vx, 0.55)
        self.assertEqual(vy, 0.0)
        self.assertLess(duration, 0.8)

    def test_cube_servo_moves_forward_on_moderate_angle_error(self):
        vx, vy, wz, duration = _navigation_velocity_command(
            target_kind="cube",
            area=6000,
            angle=-8.8,
            arrival_area=10000,
        )
        self.assertGreater(vx, 0.0)
        self.assertEqual(vy, 0.0)
        self.assertGreater(wz, 0.0)
        self.assertLess(duration, 0.8)

    def test_cube_servo_rotates_in_place_on_large_angle_error(self):
        vx, vy, wz, duration = _navigation_velocity_command(
            target_kind="cube",
            area=6000,
            angle=24.8,
            arrival_area=10000,
        )
        self.assertEqual(vx, 0.0)
        self.assertEqual(vy, 0.0)
        self.assertLess(wz, 0.0)
        self.assertLess(duration, 0.8)

    def test_head_pitch_policy_looks_slightly_down_by_default(self):
        cube_pitch = _head_pitch_for_target("cube")
        pad_pitch = _head_pitch_for_target("pad", has_held=True)
        close_cube_pitch = _head_pitch_for_target("cube", close=True)
        held_pitch = _head_pitch_for_target("cube", held_color_check=True)

        self.assertGreater(cube_pitch, pad_pitch)
        self.assertGreater(close_cube_pitch, cube_pitch)
        self.assertGreater(held_pitch, pad_pitch)

    def test_close_target_triggers_temporary_lookdown_only_when_centered(self):
        self.assertTrue(
            _should_peek_down("cube", area=3505, angle_deg=-1.5, step=1)
        )
        self.assertFalse(
            _should_peek_down("cube", area=3505, angle_deg=18.0, step=1)
        )
        self.assertTrue(
            _should_peek_down("pad", area=10500, angle_deg=4.0, step=3)
        )
        self.assertFalse(
            _should_peek_down("pad", area=10500, angle_deg=4.0, step=1)
        )

    def test_cube_arrival_requires_centered_target(self):
        arrived = _navigation_arrived(
            target_kind="cube",
            area=21250,
            angle_deg=-12.5,
            moved_toward_target=True,
            pad_direction_confirmed=False,
            pad_forward_steps=0,
            step=4,
        )
        self.assertFalse(arrived)

    def test_cube_arrival_rejects_precontact_pick_before_pushing(self):
        arrived = _navigation_arrived(
            target_kind="cube",
            area=4100,
            angle_deg=-1.5,
            moved_toward_target=True,
            pad_direction_confirmed=False,
            pad_forward_steps=0,
            step=1,
        )
        self.assertFalse(arrived)

    def test_cube_arrival_rejects_small_precontact_blob(self):
        arrived = _navigation_arrived(
            target_kind="cube",
            area=2500,
            angle_deg=-1.5,
            moved_toward_target=True,
            pad_direction_confirmed=False,
            pad_forward_steps=0,
            step=1,
        )
        self.assertFalse(arrived)

    def test_cube_arrival_allows_large_close_look_before_pushing(self):
        arrived = _navigation_arrived(
            target_kind="cube",
            area=11278,
            angle_deg=-1.8,
            moved_toward_target=False,
            pad_direction_confirmed=False,
            pad_forward_steps=0,
            step=1,
        )
        self.assertTrue(arrived)

    def test_cube_selector_rejects_live_source_zone_floor_blob(self):
        memory = AgentMemory()
        source_zone_blob = detection(
            "green",
            area=7336,
            angle=-1.8,
            centroid=(602, 693),
            bbox=(528, 665, 149, 55),
        )
        close_source_zone_blob = detection(
            "green",
            area=11274,
            angle=-1.8,
            centroid=(597, 679),
            bbox=(518, 638, 158, 82),
        )
        real_cube = detection(
            "green",
            area=3597,
            angle=-2.0,
            centroid=(598, 304),
            bbox=(567, 272, 63, 63),
        )

        self.assertIsNone(_select_target_detection([source_zone_blob], "green", "cube", memory))
        self.assertIsNone(_select_target_detection([close_source_zone_blob], "green", "cube", memory))
        selected = _select_target_detection(
            [source_zone_blob, close_source_zone_blob, real_cube],
            "green",
            "cube",
            memory,
        )
        self.assertEqual(selected, real_cube)

    def test_cube_selector_rejects_live_wide_flat_destination_decoy(self):
        memory = AgentMemory()
        decoy = detection(
            "green",
            area=4000,
            angle=-1.8,
            centroid=(598, 600),
            bbox=(500, 580, 150, 40), # wide flat aspect
        )
        self.assertIsNone(_select_target_detection([decoy], "green", "cube", memory))

if __name__ == "__main__":
    unittest.main()
