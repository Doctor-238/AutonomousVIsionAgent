import math
import unittest
from types import SimpleNamespace

from menlo_runner.programs.project.ko.level_1_starter_ko import (
    AgentDecision,
    AgentMemory,
    Observation,
    ScannedDetection,
    _add_no_place_zone,
    _bounded_step_xy,
    _choose_local_decision,
    _estimate_detection_distance,
    _fallback_status,
    _failure_mode,
    _is_unusable_pad_xy,
    _is_probable_held_or_floor_reflection,
    _is_probable_source_a_sign,
    _landmark_seek_candidates,
    _llm_backend_config,
    _looks_like_close_source_cube,
    _looks_like_cube_candidate,
    _looks_like_pad_candidate,
    _make_pallet_certificate,
    _navigation_block_reason,
    _observation_scan_plan,
    _pad_search_motion,
    _project_detection_xy,
    _remember_pad_estimates,
    _select_next_cube,
    _scored_delta,
    _supervised_go_to_xy,
    result_summary,
    update_memory,
)


def status(x=0.0, y=0.0, yaw_deg=0.0, z=0.62):
    pose = SimpleNamespace(position=(x, y, z), yaw_deg=yaw_deg)
    return SimpleNamespace(robot=SimpleNamespace(pose=pose))


def detection(
    color="green",
    *,
    area=3600,
    angle=0.0,
    centroid=(640, 360),
    bbox=(600, 320, 70, 70),
    head_yaw=0.0,
    head_pitch=0.3,
    letter_score=0.0,
    wood_score=0.0,
    feature_ready=False,
):
    return ScannedDetection(
        color=color,
        angle_deg=angle,
        blob_area=area,
        centroid=centroid,
        bbox=bbox,
        head_yaw=head_yaw,
        head_pitch=head_pitch,
        letter_score=letter_score,
        wood_score=wood_score,
        feature_ready=feature_ready,
    )


def observation(*detections):
    return Observation(robot_status=status(), detections=list(detections))


class Level1FastPolicyTest(unittest.TestCase):
    def test_projection_uses_robot_yaw_and_head_yaw(self):
        robot = status(x=1.0, y=2.0, yaw_deg=90.0)
        det = detection(angle=0.0, head_yaw=0.0, area=10000)

        x, y = _project_detection_xy(robot, det, target_kind="cube")

        self.assertAlmostEqual(x, 1.0, delta=0.05)
        self.assertGreater(y, 2.4)

    def test_cube_candidate_rejects_top_sign_shape(self):
        top_sign = detection(
            "green",
            area=5000,
            centroid=(760, 80),
            bbox=(700, 25, 120, 90),
        )
        cube = detection(
            "green",
            area=3600,
            centroid=(620, 330),
            bbox=(580, 290, 80, 80),
        )

        self.assertFalse(_looks_like_cube_candidate(top_sign))
        self.assertTrue(_looks_like_cube_candidate(cube))

    def test_tiny_far_cube_fragment_is_not_coordinate_target(self):
        tiny_fragment = detection(
            "blue",
            area=553,
            centroid=(995, 676),
            bbox=(972, 645, 43, 55),
        )

        self.assertFalse(_looks_like_cube_candidate(tiny_fragment))
        self.assertTrue(_looks_like_close_source_cube(tiny_fragment))

    def test_pad_candidate_accepts_square_mid_sign(self):
        pad_sign = detection(
            "blue",
            area=5400,
            centroid=(770, 145),
            bbox=(720, 95, 100, 90),
        )

        self.assertTrue(_looks_like_pad_candidate(pad_sign))

    def test_feature_ready_pad_rejects_plain_colored_blob(self):
        plain_blob = detection(
            "blue",
            area=5400,
            centroid=(770, 145),
            bbox=(720, 95, 100, 90),
            feature_ready=True,
            letter_score=0.0,
            wood_score=0.0,
        )

        self.assertFalse(_looks_like_pad_candidate(plain_blob))

    def test_feature_ready_pad_requires_letter_and_wood_signature(self):
        letter_pad = detection(
            "blue",
            area=5400,
            centroid=(770, 145),
            bbox=(720, 95, 100, 90),
            feature_ready=True,
            letter_score=0.03,
        )
        wood_pad = detection(
            "blue",
            area=5400,
            centroid=(770, 145),
            bbox=(720, 95, 100, 90),
            feature_ready=True,
            wood_score=0.04,
        )
        complete_pad = detection(
            "blue",
            area=5400,
            centroid=(770, 145),
            bbox=(720, 95, 100, 90),
            feature_ready=True,
            letter_score=0.12,
            wood_score=0.04,
        )

        self.assertFalse(_looks_like_pad_candidate(letter_pad))
        self.assertFalse(_looks_like_pad_candidate(wood_pad))
        self.assertTrue(_looks_like_pad_candidate(complete_pad))

    def test_live_false_blue_shelf_fragment_is_rejected(self):
        false_blue = detection(
            "blue",
            area=4164,
            centroid=(831, 328),
            bbox=(802, 292, 60, 84),
            angle=-34.0,
            feature_ready=True,
            letter_score=0.09,
            wood_score=0.063,
        )

        self.assertFalse(_looks_like_pad_candidate(false_blue))

    def test_live_large_blue_d_sign_is_landmark_not_actionable_pad(self):
        blue_d = detection(
            "blue",
            area=28143,
            centroid=(1179, 409),
            bbox=(1057, 300, 223, 242),
            feature_ready=True,
            letter_score=0.0,
            wood_score=0.001,
        )

        self.assertFalse(_looks_like_pad_candidate(blue_d))

    def test_live_blue_d_wood_pallet_candidate_is_actionable(self):
        blue_pallet = detection(
            "blue",
            area=3255,
            centroid=(1115, 60),
            bbox=(1083, 21, 80, 71),
            angle=22.3,
            feature_ready=True,
            letter_score=0.0,
            wood_score=0.512,
        )

        self.assertTrue(_looks_like_pad_candidate(blue_pallet))

    def test_top_clipped_blue_pallet_fragment_is_not_actionable(self):
        clipped = detection(
            "blue",
            area=3490,
            centroid=(1070, 30),
            bbox=(1038, 0, 68, 63),
            angle=20.2,
            feature_ready=True,
            letter_score=0.0,
            wood_score=0.9,
        )

        self.assertFalse(_looks_like_pad_candidate(clipped))

    def test_side_clipped_blue_d_pallet_fragment_can_be_actionable(self):
        side_visible_pallet = detection(
            "blue",
            area=3043,
            centroid=(65, 63),
            bbox=(0, 24, 103, 84),
            angle=-29.1,
            feature_ready=True,
            letter_score=0.0,
            wood_score=0.851,
        )

        self.assertTrue(_looks_like_pad_candidate(side_visible_pallet))

    def test_green_c_pad_allows_lower_start_view_signature(self):
        green_c = detection(
            "green",
            area=5200,
            centroid=(95, 338),
            bbox=(42, 285, 92, 96),
            feature_ready=True,
            letter_score=0.065,
            wood_score=0.018,
        )

        self.assertTrue(_looks_like_pad_candidate(green_c))

    def test_local_decision_projects_cube_coordinate(self):
        memory = AgentMemory()
        obs = observation(
            detection("red", area=3300, angle=-4.0, centroid=(580, 340), bbox=(540, 300, 75, 75))
        )

        decision = _choose_local_decision(obs, memory, None)

        self.assertEqual(decision.next_action, "navigate_to_cube")
        self.assertEqual(decision.target_color, "red")
        self.assertIsNotNone(memory.active_target_xy)
        self.assertTrue(all(math.isfinite(value) for value in memory.active_target_xy))

    def test_close_source_cube_triggers_direct_pick_before_navigation(self):
        memory = AgentMemory()
        obs = observation(
            detection("blue", area=553, angle=16.0, centroid=(995, 676), bbox=(972, 645, 43, 55))
        )

        decision = _choose_local_decision(obs, memory, None)

        self.assertEqual(decision.next_action, "pick_cube")
        self.assertEqual(memory.active_target_kind, "source")

    def test_local_decision_goes_to_pad_when_holding_cube(self):
        memory = AgentMemory(held_color="yellow", stage="need_pad", known_pad_xy={"yellow": (2.0, -1.0)})
        decision = _choose_local_decision(observation(), memory, None)

        self.assertEqual(decision.next_action, "navigate_to_pad")
        self.assertEqual(decision.target_color, "yellow")

    def test_local_decision_searches_pad_when_holding_without_coordinate(self):
        memory = AgentMemory(held_color="yellow", stage="need_pad")
        decision = _choose_local_decision(observation(), memory, None)

        self.assertEqual(decision.next_action, "search_pad")
        self.assertEqual(decision.target_color, "yellow")

    def test_cube_distance_is_bounded_for_source_belt(self):
        det = detection("blue", area=3600, centroid=(580, 340), bbox=(540, 300, 75, 75))

        distance = _estimate_detection_distance(det, target_kind="cube")

        self.assertGreaterEqual(distance, 0.45)
        self.assertLess(distance, 1.35)

    def test_pad_distance_keeps_standoff_from_shelving(self):
        det = detection("blue", area=5400, centroid=(770, 145), bbox=(720, 95, 100, 90))

        distance = _estimate_detection_distance(det, target_kind="pad")

        self.assertGreaterEqual(distance, 0.85)
        self.assertLessEqual(distance, 2.35)

    def test_far_blue_pallet_ray_projects_deeper_toward_d(self):
        det = detection(
            "blue",
            area=1949,
            centroid=(881, 101),
            bbox=(829, 80, 90, 45),
            feature_ready=True,
            wood_score=0.697,
        )

        distance = _estimate_detection_distance(det, target_kind="pad")

        self.assertGreaterEqual(distance, 2.75)

    def test_large_blue_d_sign_projects_deeper_than_generic_standoff(self):
        det = detection("blue", area=11687, centroid=(1156, 387), bbox=(1060, 338, 192, 105))

        distance = _estimate_detection_distance(det, target_kind="pad")

        self.assertGreater(distance, 1.35)

    def test_bounded_step_splits_long_pad_navigation(self):
        nav_xy, partial = _bounded_step_xy((0.0, 0.0), (4.0, 0.0), max_step_m=2.1)

        self.assertTrue(partial)
        self.assertAlmostEqual(nav_xy[0], 2.1)
        self.assertAlmostEqual(nav_xy[1], 0.0)

    def test_pad_search_motion_uses_wide_then_exploratory_pattern(self):
        memory = AgentMemory()

        first_motion, first_status = _pad_search_motion(memory)
        second_motion, second_status = _pad_search_motion(memory)

        self.assertEqual(first_status, "rotate_about_180")
        self.assertGreaterEqual(first_motion["duration_s"], 4.0)
        self.assertEqual(second_status, "back_left_rescan")
        self.assertLess(second_motion["duration_s"], 1.5)

    def test_scan_plan_skips_when_pad_or_source_coordinate_is_known(self):
        self.assertIsNone(_observation_scan_plan(AgentMemory(stage="ready_place", held_color="green")))
        self.assertIsNone(
            _observation_scan_plan(
                AgentMemory(held_color="green", stage="need_pad", known_pad_xy={"green": (1.0, 1.0)})
            )
        )
        self.assertIsNone(_observation_scan_plan(AgentMemory(known_source_xy=(0.0, 0.0), stage="need_cube")))
        self.assertIsNotNone(_observation_scan_plan(AgentMemory(held_color="green", stage="need_pad")))

    def test_priority_color_does_not_override_much_better_cube(self):
        memory = AgentMemory(priority_colors=["blue"])
        obs = observation(
            detection("blue", area=900, angle=28.0, centroid=(820, 180), bbox=(800, 155, 34, 34)),
            detection("red", area=5000, angle=0.0, centroid=(640, 420), bbox=(600, 380, 85, 85)),
        )

        target = _select_next_cube(obs, memory)

        self.assertEqual(target.color, "red")

    def test_pick_failure_clears_bad_target_and_temporarily_blocks_color(self):
        memory = AgentMemory(
            active_color="blue",
            active_target_xy=(4.2, -0.3),
            active_target_kind="cube",
            stage="ready_pick",
        )
        verified = {
            "action_result": {"action": "pick_cube", "ok": False, "error": "too far"},
            "held_color": None,
            "delivered_count": 0,
        }

        update_memory(memory, observation(), AgentDecision("pick_cube", "blue"), verified)

        self.assertIsNone(memory.active_color)
        self.assertIsNone(memory.active_target_xy)
        self.assertEqual(memory.blocked_colors["blue"], 4)

    def test_returns_to_remembered_source_before_hunting_distant_blobs(self):
        memory = AgentMemory(known_source_xy=(3.0, 3.0), stage="need_cube")
        obs = Observation(robot_status=status(0.0, 0.0), detections=[])

        decision = _choose_local_decision(obs, memory, None)

        self.assertEqual(decision.next_action, "navigate_to_cube")
        self.assertIsNone(decision.target_color)
        self.assertEqual(memory.active_target_kind, "source")
        self.assertEqual(memory.active_target_xy, (3.0, 3.0))

    def test_near_source_anchor_picks_instead_of_slow_return_navigation(self):
        memory = AgentMemory(known_source_xy=(0.0, 0.0), stage="need_cube")
        obs = Observation(
            robot_status=status(0.75, 0.0),
            detections=[detection("green", area=6000, centroid=(700, 650), bbox=(640, 590, 110, 90))],
        )

        decision = _choose_local_decision(obs, memory, None)

        self.assertEqual(decision.next_action, "pick_cube")
        self.assertEqual(memory.active_target_kind, "source")
        self.assertEqual(memory.active_target_xy, (0.0, 0.0))

    def test_fallen_robot_stops_instead_of_repeating_actions(self):
        memory = AgentMemory(fallen_detected=True)

        decision = _choose_local_decision(observation(), memory, None)

        self.assertEqual(decision.next_action, "stop")

    def test_unavailable_robot_status_does_not_navigate_on_fake_origin(self):
        memory = AgentMemory()
        obs = Observation(
            robot_status=_fallback_status(),
            detections=[detection("green", area=6000, centroid=(700, 650), bbox=(640, 590, 110, 90))],
        )

        decision = _choose_local_decision(obs, memory, None)

        self.assertEqual(decision.next_action, "search_cube")

    def test_picks_generically_after_returning_to_source_anchor(self):
        memory = AgentMemory(stage="ready_pick", active_target_kind="source")

        decision = _choose_local_decision(observation(), memory, None)

        self.assertEqual(decision.next_action, "pick_cube")
        self.assertIsNone(decision.target_color)

    def test_source_anchor_direct_pick_when_nearby(self):
        memory = AgentMemory(known_source_xy=(0.0, 0.0), stage="need_cube")
        obs = Observation(robot_status=status(0.25, 0.1), detections=[])

        decision = _choose_local_decision(obs, memory, None)

        self.assertEqual(decision.next_action, "pick_cube")
        self.assertEqual(memory.active_target_kind, "source")
        self.assertEqual(memory.active_target_xy, (0.0, 0.0))

    def test_source_pick_cooldown_recovers_before_retrying(self):
        memory = AgentMemory(known_source_xy=(0.0, 0.0), source_pick_cooldown=1, stage="need_cube")
        obs = Observation(robot_status=status(0.2, 0.0), detections=[])

        decision = _choose_local_decision(obs, memory, None)

        self.assertEqual(decision.next_action, "recover")
        self.assertEqual(decision.recovery_strategy, "source_pick_nudge")
        self.assertEqual(memory.source_pick_cooldown, 0)

    def test_held_color_overrides_original_pick_target_for_delivery(self):
        memory = AgentMemory(
            held_color="blue",
            active_color="red",
            stage="need_pad",
            known_pad_xy={"blue": (2.0, -1.0)},
        )

        decision = _choose_local_decision(observation(), memory, None)

        self.assertEqual(decision.next_action, "navigate_to_pad")
        self.assertEqual(decision.target_color, "blue")
        self.assertEqual(memory.active_color, "blue")

    def test_blocked_held_color_discards_instead_of_repeating_bad_pad_search(self):
        memory = AgentMemory(
            held_color="blue",
            active_color="blue",
            stage="need_pad",
            blocked_colors={"blue": 6},
        )

        decision = _choose_local_decision(observation(), memory, None)

        self.assertEqual(decision.next_action, "recover")
        self.assertEqual(decision.target_color, "blue")
        self.assertEqual(decision.recovery_strategy, "discard_held_at_source")

    def test_successful_pick_records_robot_position_as_source_anchor(self):
        memory = AgentMemory(
            active_color="blue",
            active_target_xy=(4.2, -0.3),
            active_target_kind="cube",
            stage="ready_pick",
        )
        verified = {
            "action_result": {"action": "pick_cube", "ok": True},
            "held_color": "blue",
            "delivered_count": 0,
            "robot_xy": (1.1, -1.7),
        }

        update_memory(memory, observation(), AgentDecision("pick_cube", "blue"), verified)

        self.assertEqual(memory.known_source_xy, (1.1, -1.7))

    def test_successful_place_confirms_pad_coordinate(self):
        memory = AgentMemory(target_pad_xy=(2.1, -2.4), stage="ready_place")
        verified = {
            "action_result": {"action": "place_cube", "ok": True, "was_holding": True},
            "held_color": None,
            "delivered_count": 1,
        }

        update_memory(memory, observation(), AgentDecision("place_cube", "green"), verified)

        self.assertEqual(memory.confirmed_pad_xy["green"], (2.1, -2.4))
        self.assertEqual(memory.known_pad_xy["green"], (2.1, -2.4))

    def test_failed_place_rejects_bad_pad_coordinate(self):
        memory = AgentMemory(target_pad_xy=(-2.1, -0.5), stage="ready_place")
        verified = {
            "action_result": {"action": "place_cube", "ok": False, "was_holding": True},
            "held_color": "green",
            "delivered_count": 0,
        }

        update_memory(memory, observation(), AgentDecision("place_cube", "green"), verified)

        self.assertEqual(memory.rejected_pad_xy["green"], [(-2.1, -0.5)])
        self.assertNotIn("green", memory.known_pad_xy)

    def test_place_without_score_does_not_confirm_pad(self):
        memory = AgentMemory(target_pad_xy=(1.2, -1.0), stage="ready_place", delivered_count=0)
        verified = {
            "action_result": {"action": "place_cube", "ok": True, "was_holding": True},
            "held_color": None,
            "delivered_count": 0,
        }

        update_memory(memory, observation(), AgentDecision("place_cube", "blue"), verified)

        self.assertNotIn("blue", memory.confirmed_pad_xy)
        self.assertEqual(memory.rejected_pad_xy["blue"], [(1.2, -1.0)])
        self.assertTrue(any(zone["reason"] == "place_without_score" for zone in memory.no_place_zones))

    def test_failed_blue_place_rejects_coordinate_without_blocking_remap(self):
        memory = AgentMemory(target_pad_xy=(2.1, -0.7), stage="ready_place", held_color="blue", delivered_count=0)
        verified = {
            "action_result": {"action": "place_cube", "ok": False, "was_holding": True},
            "held_color": "blue",
            "delivered_count": 0,
        }

        update_memory(memory, observation(), AgentDecision("place_cube", "blue"), verified)

        self.assertNotIn("blue", memory.blocked_colors)
        self.assertEqual(memory.rejected_pad_xy["blue"], [(2.1, -0.7)])
        self.assertEqual(memory.stage, "need_pad")

    def test_baseline_delivered_count_tracks_scored_delta(self):
        memory = AgentMemory()

        update_memory(
            memory,
            observation(),
            AgentDecision("search_cube"),
            {"action_result": {"action": "search_cube", "ok": True}, "held_color": None, "delivered_count": 5},
        )
        self.assertEqual(_scored_delta(memory), 0)

        update_memory(
            memory,
            observation(),
            AgentDecision("place_cube", "green"),
            {
                "action_result": {"action": "place_cube", "ok": True, "was_holding": True},
                "held_color": None,
                "delivered_count": 6,
            },
        )
        self.assertEqual(_scored_delta(memory), 1)

    def test_missing_progress_count_does_not_poison_baseline(self):
        memory = AgentMemory()

        update_memory(
            memory,
            observation(),
            AgentDecision("search_cube"),
            {"action_result": {"action": "search_cube", "ok": True}, "held_color": None},
        )

        self.assertIsNone(memory.baseline_delivered_count)
        self.assertEqual(memory.delivered_count, 0)

    def test_partial_pad_navigation_does_not_place_immediately(self):
        memory = AgentMemory(target_pad_xy=(3.0, -1.0), known_pad_xy={"blue": (3.0, -1.0)}, stage="need_pad")
        verified = {
            "action_result": {"action": "navigate_to_pad", "ok": True, "partial": True},
            "held_color": "blue",
            "delivered_count": 0,
        }

        update_memory(memory, observation(), AgentDecision("navigate_to_pad", "blue"), verified)

        self.assertEqual(memory.stage, "need_pad")
        self.assertFalse(memory.pad_ready)

    def test_failed_pad_navigation_rejects_bad_pad_coordinate(self):
        memory = AgentMemory(
            target_pad_xy=(4.7, -3.4),
            known_pad_xy={"green": (4.7, -3.4)},
            stage="need_pad",
        )
        verified = {
            "action_result": {"action": "navigate_to_pad", "ok": False, "result": {"error": "stuck"}},
            "held_color": "green",
            "delivered_count": 0,
        }

        update_memory(memory, observation(), AgentDecision("navigate_to_pad", "green"), verified)

        self.assertEqual(memory.rejected_pad_xy["green"], [(4.7, -3.4)])
        self.assertNotIn("green", memory.known_pad_xy)
        self.assertIsNone(memory.target_pad_xy)

    def test_preflight_block_does_not_reject_pad_coordinate(self):
        memory = AgentMemory(
            target_pad_xy=(4.4, -0.35),
            known_pad_xy={"green": (4.4, -0.35)},
            stage="need_pad",
        )
        verified = {
            "action_result": {"action": "navigate_to_pad", "ok": False, "reason": "no_place_zone"},
            "held_color": "green",
            "delivered_count": 0,
        }

        update_memory(memory, observation(), AgentDecision("navigate_to_pad", "green"), verified)

        self.assertNotIn("green", memory.rejected_pad_xy)
        self.assertNotIn("green", memory.known_pad_xy)
        self.assertIsNone(memory.target_pad_xy)

    def test_preflight_block_temporarily_blocks_repeated_pad_candidate(self):
        blocked_xy = (0.97, 0.67)
        memory = AgentMemory(
            target_pad_xy=blocked_xy,
            known_pad_xy={"blue": blocked_xy},
            held_color="blue",
            stage="need_pad",
        )
        verified = {
            "action_result": {"action": "navigate_to_pad", "ok": False, "reason": "no_place_zone", "target_xy": blocked_xy},
            "held_color": "blue",
            "delivered_count": 0,
        }

        update_memory(memory, observation(), AgentDecision("navigate_to_pad", "blue"), verified)

        self.assertNotIn("blue", memory.rejected_pad_xy)
        self.assertNotIn("blue", memory.known_pad_xy)
        self.assertTrue(_is_unusable_pad_xy(memory, "blue", blocked_xy))

    def test_real_pad_can_be_near_source_anchor(self):
        memory = AgentMemory(known_source_xy=(1.0, -1.4))

        self.assertFalse(_is_unusable_pad_xy(memory, "green", (1.7, -1.4)))
        self.assertFalse(_is_unusable_pad_xy(memory, "green", (1.3, -1.4)))
        self.assertFalse(_is_unusable_pad_xy(memory, "blue", (1.7, -1.4)))
        self.assertTrue(_is_unusable_pad_xy(memory, "blue", (1.3, -1.4)))
        self.assertTrue(_is_unusable_pad_xy(memory, "green", (1.1, -1.35)))

    def test_other_confirmed_pad_area_is_unusable_for_new_color(self):
        memory = AgentMemory(confirmed_pad_xy={"green": (2.31, -2.51)})

        self.assertTrue(_is_unusable_pad_xy(memory, "blue", (2.50, -2.62)))
        self.assertFalse(_is_unusable_pad_xy(memory, "green", (2.50, -2.62)))
        self.assertFalse(_is_unusable_pad_xy(memory, "blue", (3.45, -2.62)))

    def test_rejected_pad_coordinate_is_not_remembered_again(self):
        obs = observation(
            detection("green", area=5400, angle=0.0, centroid=(640, 145), bbox=(600, 105, 80, 80))
        )
        bad_xy = _project_detection_xy(obs.robot_status, obs.detections[0], target_kind="pad")
        memory = AgentMemory(
            rejected_pad_xy={"green": [bad_xy]},
            held_color="green",
            stage="need_pad",
        )

        _remember_pad_estimates(obs, memory)

        self.assertNotIn("green", memory.known_pad_xy)

    def test_unavailable_status_does_not_remember_pad_coordinate(self):
        obs = Observation(
            robot_status=_fallback_status(),
            detections=[
                detection(
                    "green",
                    area=5400,
                    angle=0.0,
                    centroid=(640, 145),
                    bbox=(600, 105, 80, 80),
                    feature_ready=True,
                    letter_score=0.03,
                )
            ],
        )
        memory = AgentMemory(held_color="green", stage="need_pad")

        _remember_pad_estimates(obs, memory)

        self.assertNotIn("green", memory.known_pad_xy)

    def test_non_held_color_is_not_remembered_as_current_pad_target(self):
        memory = AgentMemory(held_color="green", stage="need_pad")
        obs = Observation(
            robot_status=status(),
            detections=[
                detection(
                    "green",
                    area=5200,
                    centroid=(95, 338),
                    bbox=(42, 285, 92, 96),
                    angle=-24.0,
                    feature_ready=True,
                    letter_score=0.08,
                    wood_score=0.08,
                ),
                detection(
                    "blue",
                    area=3255,
                    centroid=(1115, 60),
                    bbox=(1083, 21, 80, 71),
                    angle=22.3,
                    feature_ready=True,
                    letter_score=0.0,
                    wood_score=0.512,
                ),
            ],
        )

        _remember_pad_estimates(obs, memory)

        self.assertIn("green", memory.known_pad_xy)
        self.assertNotIn("blue", memory.known_pad_xy)

    def test_live_green_a_source_sign_is_blocked_but_left_c_is_allowed(self):
        memory = AgentMemory(known_source_xy=(0.406, -1.497), held_color="green", stage="need_pad")
        robot_status = status(x=0.406, y=-1.497, yaw_deg=0.0)
        a_sign = detection(
            "green",
            area=4481,
            centroid=(413, 299),
            bbox=(379, 265, 70, 71),
            angle=15.2,
            feature_ready=True,
            letter_score=0.087,
            wood_score=0.139,
        )
        c_sign = detection(
            "green",
            area=8040,
            centroid=(83, 352),
            bbox=(38, 307, 93, 92),
            angle=-26.1,
            feature_ready=True,
            letter_score=0.089,
            wood_score=0.142,
        )

        a_xy = _project_detection_xy(robot_status, a_sign, target_kind="pad")
        c_xy = _project_detection_xy(robot_status, c_sign, target_kind="pad")

        self.assertTrue(_is_probable_source_a_sign(memory, robot_status, a_sign, a_xy))
        self.assertFalse(_is_probable_source_a_sign(memory, robot_status, c_sign, c_xy))

    def test_live_green_right_side_source_false_positive_is_not_remembered(self):
        memory = AgentMemory(known_source_xy=(0.406, -1.497), held_color="green", stage="need_pad")
        robot_status = status(x=0.406, y=-1.497, yaw_deg=0.0)
        false_right = detection(
            "green",
            area=4907,
            centroid=(907, 257),
            bbox=(872, 220, 72, 75),
            angle=87.5,
            feature_ready=True,
            letter_score=0.092,
            wood_score=0.121,
        )
        obs = Observation(robot_status=robot_status, detections=[false_right])

        _remember_pad_estimates(obs, memory)

        self.assertNotIn("green", memory.known_pad_xy)

    def test_live_held_cube_floor_reflection_is_not_place_evidence(self):
        memory = AgentMemory(target_pad_xy=(0.48, 0.43), held_color="green", stage="ready_place")
        robot_status = status(x=0.59, y=-0.27, yaw_deg=0.0)
        reflected_held_cube = detection(
            "green",
            area=13800,
            centroid=(735, 515),
            bbox=(668, 445, 164, 118),
            angle=5.2,
            feature_ready=True,
            letter_score=0.08,
            wood_score=0.13,
        )
        obs = Observation(robot_status=robot_status, detections=[reflected_held_cube])

        reflected_xy = _project_detection_xy(robot_status, reflected_held_cube, target_kind="pad")

        self.assertTrue(_is_probable_held_or_floor_reflection(memory, robot_status, reflected_held_cube, reflected_xy))
        self.assertIsNone(_make_pallet_certificate(obs, memory, "green", target_xy=(0.48, 0.43)))

    def test_landmark_seek_uses_close_c_sign_but_not_huge_a_sign(self):
        memory = AgentMemory(known_source_xy=(0.406, -1.497), held_color="green", stage="need_pad")
        robot_status = status(x=1.87, y=-1.74, yaw_deg=0.0)
        c_sign = detection(
            "green",
            area=12019,
            centroid=(62, 209),
            bbox=(7, 73, 112, 208),
            angle=-27.1,
            feature_ready=True,
            letter_score=0.575,
            wood_score=0.0,
        )
        huge_a = detection(
            "green",
            area=240000,
            centroid=(850, 350),
            bbox=(451, 0, 829, 720),
            angle=6.5,
            feature_ready=True,
            letter_score=0.15,
            wood_score=0.0,
        )
        obs = Observation(robot_status=robot_status, detections=[huge_a, c_sign])

        candidates = _landmark_seek_candidates(obs, memory, "green")

        self.assertEqual(candidates, [c_sign])

    def test_no_place_zone_blocks_pad_navigation_target(self):
        memory = AgentMemory()
        _add_no_place_zone(memory, (1.0, -1.0), reason="source_conveyor_A", radius=0.6)

        reason = _navigation_block_reason(memory, (1.2, -1.05), purpose="pad", color="blue")

        self.assertEqual(reason, "no_place_zone")

    def test_near_pallet_certificate_requires_close_visual_evidence(self):
        memory = AgentMemory(target_pad_xy=(1.0, -1.0), held_color="blue", stage="ready_place")
        obs_without_pallet = Observation(robot_status=status(x=1.1, y=-1.05), detections=[])

        self.assertIsNone(_make_pallet_certificate(obs_without_pallet, memory, "blue", target_xy=(1.0, -1.0)))

        obs_with_pallet = Observation(
            robot_status=status(x=1.1, y=-1.05),
            detections=[
                detection(
                    "blue",
                    area=3255,
                    centroid=(1115, 60),
                    bbox=(1083, 21, 80, 71),
                    feature_ready=True,
                    wood_score=0.512,
                )
            ],
        )

        cert = _make_pallet_certificate(obs_with_pallet, memory, "blue", target_xy=(1.0, -1.0))

        self.assertIsNotNone(cert)
        self.assertEqual(cert["color"], "blue")

    def test_tokamak_backend_takes_precedence_over_openrouter(self):
        url, key, model = _llm_backend_config(
            {
                "TOKAMAK_API_KEY": "tokamak-secret",
                "TOKAMAK_MODEL": "qwen/qwen3.6-35b-a3b",
                "OPENROUTER_API_KEY": "openrouter-secret",
                "OPENROUTER_MODEL": "openrouter/free",
            }
        )

        self.assertIn("tokamak.sh", url)
        self.assertEqual(key, "tokamak-secret")
        self.assertEqual(model, "qwen/qwen3.6-35b-a3b")

    def test_confirmed_pad_coordinate_is_not_overwritten_by_source_view(self):
        memory = AgentMemory(
            confirmed_pad_xy={"green": (2.1, -2.4)},
            known_pad_xy={"green": (2.1, -2.4)},
            held_color="green",
            stage="need_pad",
        )
        obs = observation(
            detection("green", area=5000, angle=0.0, centroid=(640, 340), bbox=(600, 300, 85, 85))
        )

        _remember_pad_estimates(obs, memory)

        self.assertEqual(memory.known_pad_xy["green"], (2.1, -2.4))

    def test_result_summary_preserves_unicode_sdk_errors(self):
        result = SimpleNamespace(
            status="failed",
            error=SimpleNamespace(message="path failed — not reachable"),
        )

        summary = result_summary(result)

        self.assertEqual(summary["error"], "path failed — not reachable")


    def test_failure_mode_classifies_common_failures(self):
        self.assertEqual(_failure_mode({"action": "place_cube", "ok": False, "error": "not near pallet"}), "place_verify")
        self.assertEqual(_failure_mode({"action": "pick_cube", "ok": False, "error": "No visible cubes"}), "source_pick")
        self.assertEqual(_failure_mode({"action": "navigate_to_pad", "ok": False, "error": "robot is fallen"}), "fallen")


class FakeNavigationContext:
    def __init__(self, robot_status, *, invoke_exc=None, invoke_status="done", invoke_error=None):
        self.robot_status = robot_status
        self.invoke_exc = invoke_exc
        self.invoke_status = invoke_status
        self.invoke_error = invoke_error
        self.cancel_called = False

    async def invoke(self, name, args=None, timeout_s=None):
        if name == "cancel":
            self.cancel_called = True
            return SimpleNamespace(status="done", error=None)
        if name == "go_to" and self.invoke_exc is not None:
            raise self.invoke_exc
        error = SimpleNamespace(message=self.invoke_error) if self.invoke_error else None
        return SimpleNamespace(status=self.invoke_status, error=error)

    async def state(self, name):
        return self.robot_status


class Level1NavigationSupervisorTest(unittest.IsolatedAsyncioTestCase):
    async def test_supervised_go_to_treats_timeout_as_success_when_robot_arrived(self):
        memory = AgentMemory()
        ctx = FakeNavigationContext(status(0.1, 0.1), invoke_exc=TimeoutError("late reply"))

        result = await _supervised_go_to_xy(
            ctx,
            memory,
            (0.0, 0.0),
            action="navigate_to_cube",
            tolerance_m=0.5,
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["corrected_by_position"])
        self.assertFalse(ctx.cancel_called)

    async def test_supervised_go_to_cancels_timeout_when_robot_far(self):
        memory = AgentMemory()
        ctx = FakeNavigationContext(status(3.0, 3.0), invoke_exc=TimeoutError("late reply"))

        result = await _supervised_go_to_xy(
            ctx,
            memory,
            (0.0, 0.0),
            action="navigate_to_cube",
            tolerance_m=0.5,
        )

        self.assertFalse(result["ok"])
        self.assertTrue(ctx.cancel_called)

    async def test_pad_navigation_stuck_is_not_corrected_by_position(self):
        memory = AgentMemory()
        ctx = FakeNavigationContext(
            status(0.2, 0.2),
            invoke_status="failed",
            invoke_error="robot stuck for 5000ms",
        )

        result = await _supervised_go_to_xy(
            ctx,
            memory,
            (0.0, 0.0),
            action="navigate_to_pad",
            tolerance_m=0.5,
        )

        self.assertFalse(result["ok"])
        self.assertFalse(result["corrected_by_position"])


if __name__ == "__main__":
    unittest.main()
