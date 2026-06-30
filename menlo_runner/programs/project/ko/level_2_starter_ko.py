from __future__ import annotations

"""Level 2 프로젝트 스타터입니다.

이 파일은 완성된 해답이 아니라 최소 scaffold입니다.

SUPPORT CODE 영역은 반복해서 작성할 필요가 없는 wrapper, 자료 구조,
schema validation을 제공합니다. STUDENT TODO 영역은 팀이 직접 설계하고,
개선하고, 테스트하고, 발표에서 설명해야 하는 부분입니다.

Level 2 규칙: `scene_state`, 정확한 entity ID, coordinate `go_to`를 사용할 수
없습니다. 카메라 관찰값, `set_head`, `set_velocity`, memory, recovery로
navigation을 구현해야 합니다.
"""

import asyncio
import base64
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from menlo_runner.llm import ask_vlm
from menlo_runner.perception import detect_color_blobs


# ---------------------------------------------------------------------------
# SUPPORT CODE: 공통 과제 정의와 필수 LLM decision schema
# ---------------------------------------------------------------------------
TASK = "Find and sort the six cubes in the warehouse into their matching destination pads."

DESTINATION_SIGN_RULES = {
    "red": "B",
    "green": "C",
    "blue": "D",
    "yellow": "E",
}
COLOR_ORDER = ("red", "green", "blue", "yellow")
DEFAULT_DELIVERY_LIMIT = 6
PICK_BLOB_AREA = 10000
PLACE_BLOB_AREA = 5000
CUBE_PRECONTACT_PICK_AREA = 3200
CUBE_APPROACH_PICK_AREA = 5000
# 큐브 자체는 대개 5000~25000 px, 그 이상은 벽/바닥 패치 가능성 높음
MAX_CUBE_BLOB_AREA = 30000
# 패드 표지판은 대개 3000~15000 px, 그 이상은 벽/장애물 가능성 높음
MAX_PAD_TRACK_BLOB_AREA = 15000
CUBE_CENTERED_DEG = 10.0
PAD_CENTERED_DEG = 10.0
CUBE_IMMEDIATE_PICK_AREA = PICK_BLOB_AREA * 2.5
CUBE_STAGNANT_AREA_DELTA = 250
CUBE_STAGNANT_STEP_LIMIT = 3
HEAD_POSE_EPSILON = 0.01
LLM_DECISION_TIMEOUT_S = 4
LLM_DECISION_MAX_TOKENS = 160
ROBOT_STATUS_TIMEOUT_S = 8.0
HEAD_RPC_TIMEOUT_S = 8.0
MOVEMENT_RPC_TIMEOUT_S = 8.0
MANIPULATION_RPC_TIMEOUT_S = 45.0
MAX_FAILED_ATTEMPTS_PER_COLOR = 3
RECENT_OUTCOME_LIMIT = 8
USE_VLM_PAD_HINTS = os.environ.get("MENLO_USE_VLM_HINTS", "").lower() in {"1", "true", "yes"}
SAVE_POV_FRAMES = os.environ.get("MENLO_SAVE_POV", "").lower() in {"1", "true", "yes"}
POV_FRAME_LIMIT = max(0, int(os.environ.get("MENLO_POV_FRAME_LIMIT", "120")))
POV_FRAME_EVERY = max(1, int(os.environ.get("MENLO_POV_FRAME_EVERY", "1")))
POV_FRAME_DIR = os.environ.get("MENLO_POV_FRAME_DIR", os.path.join("outputs", "pov"))
COLOR_ALIASES = {
    "red": ("red", "빨간", "빨강", "적색"),
    "green": ("green", "초록", "녹색"),
    "blue": ("blue", "파란", "파랑", "청색"),
    "yellow": ("yellow", "노란", "노랑", "황색"),
}
SIGNAGE_NOTE = (
    "A is the conveyor/cube source area, not a destination. "
    "Destination signs are B red, C green, D blue, E yellow."
)

ALLOWED_NEXT_ACTIONS = {
    "search_cube",
    "navigate_to_cube",
    "pick_cube",
    "search_pad",
    "navigate_to_pad",
    "place_cube",
    "recover",
    "skip_target",
    "stop",
}


def get_task_instruction() -> str:
    """Allow hidden evaluation variants without source edits."""
    return (
        os.environ.get("MENLO_TASK")
        or os.environ.get("PROJECT_TASK")
        or os.environ.get("ROBOT_TASK")
        or TASK
    )


@dataclass
class AgentDecision:
    """LLM이 반환하고 코드가 검증한 고수준 decision입니다."""

    next_action: str
    target_color: str | None = None
    reason: str = ""
    recovery_strategy: str | None = None


@dataclass
class AgentMemory:
    """observe-decide-act cycle 사이에 유지하는 agent 상태입니다."""

    delivered_count: int = 0
    delivery_limit: int | None = None
    priority_colors: list[str] = field(default_factory=list)
    held_color: str | None = None
    active_color: str | None = None
    stage: str = "need_cube"
    cube_ready: bool = False
    pad_ready: bool = False
    head_yaw: float | None = None
    head_pitch: float | None = None
    nav_track_kind: str | None = None
    nav_track_color: str | None = None
    nav_track_angle: float | None = None
    nav_track_lost_steps: int = 0
    last_robot_status: Any | None = None
    robot_status_failures: int = 0
    known_pad_bearings: dict[str, dict[str, Any]] = field(default_factory=dict)
    search_turns: int = 0
    failed_attempts: dict[str, int] = field(default_factory=dict)
    completed_colors: list[str] = field(default_factory=list)
    skipped_colors: list[str] = field(default_factory=list)
    recent_outcomes: list[dict[str, Any]] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Observation:
    """LLM과 action code에 전달할 compact observation입니다."""

    robot_status: Any
    detections: list[Any]
    note: str = ""
    vlm_summary: str = ""


@dataclass
class FallbackPose:
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    yaw_deg: float = 0.0


@dataclass
class FallbackRobot:
    held_entity_ids: list[Any] = field(default_factory=list)
    pose: FallbackPose = field(default_factory=FallbackPose)


@dataclass
class FallbackRobotStatus:
    robot: FallbackRobot = field(default_factory=FallbackRobot)
    unavailable: bool = True


@dataclass(frozen=True)
class ScannedDetection:
    """head pose를 함께 저장한 색상 detection입니다."""

    color: str
    angle_deg: float
    blob_area: int
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    head_yaw: float
    head_pitch: float

    @property
    def full_bearing_deg(self) -> float:
        """image angle에 head yaw를 더한 대략적인 body-relative bearing입니다."""
        return self.angle_deg + math.degrees(self.head_yaw)


def normalize_color(value: Any) -> str | None:
    """Normalize English/Korean color words from tasks or LLM replies."""
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"", "none", "null", "any"}:
        return None

    for color, aliases in COLOR_ALIASES.items():
        if lowered == color or any(alias.lower() in lowered for alias in aliases):
            return color
    return lowered if lowered in COLOR_ORDER else None


def parse_task_instructions_local(task: str) -> tuple[int | None, list[str]]:
    """Fast local parser for the workshop task. Uses the API only as fallback."""
    delivery_limit: int | None = None

    limit_patterns = (
        r"(\d+)\s*(?:\uac1c\s*)?\ub9cc",
        r"(?:only|just|at most|max(?:imum)?|limit(?:ed)? to)\D{0,20}(\d+)",
        r"(\d+)\D{0,10}(?:only|just)",
    )
    for pattern in limit_patterns:
        match = re.search(pattern, task, flags=re.IGNORECASE)
        if match:
            delivery_limit = int(match.group(1))
            break

    lowered = task.lower()
    positions: list[tuple[int, str]] = []
    for color, aliases in COLOR_ALIASES.items():
        found = [lowered.find(alias.lower()) for alias in aliases if lowered.find(alias.lower()) >= 0]
        if found:
            positions.append((min(found), color))

    priority_colors: list[str] = []
    for _, color in sorted(positions):
        if color not in priority_colors:
            priority_colors.append(color)

    return delivery_limit, priority_colors


def _visible_detections(observation: Observation) -> list[Any]:
    return [detection for detection in observation.detections if getattr(detection, "blob_area", 0) > 0]


def _best_detection(observation: Observation, target_color: str | None = None) -> Any | None:
    candidates = _visible_detections(observation)
    if target_color:
        candidates = [detection for detection in candidates if detection.color == target_color]
    if not candidates:
        return None
    return max(candidates, key=lambda detection: detection.blob_area)


def _next_target_color(memory: AgentMemory) -> str | None:
    ordered = list(memory.priority_colors) + list(COLOR_ORDER)
    for color in ordered:
        if color not in memory.skipped_colors:
            return color
    return None


def _failed_too_often(memory: AgentMemory, color: str | None) -> bool:
    if not color:
        return False
    return memory.failed_attempts.get(color, 0) >= MAX_FAILED_ATTEMPTS_PER_COLOR


def _looks_like_held_cube_blob(detection: Any) -> bool:
    """Filter the carried cube from pad search; it appears large and low in POV."""
    x, y, width, height, cx, cy, area, aspect = _bbox_metrics(detection)
    del x
    small_center_hand_blob = (
        area >= 500
        and y >= 285
        and cy >= 320
        and 520 <= cx <= 760
        and 0.55 <= aspect <= 1.8
    )
    large_low_hand_blob = (
        area >= 8000
        and cy >= 285
        and y >= 210
        and 0.45 <= aspect <= 2.2
    )
    return small_center_hand_blob or large_low_hand_blob


def _pad_mapping_candidates(detections: list[Any]) -> list[Any]:
    return [
        detection
        for detection in detections
        if getattr(detection, "blob_area", 0) <= MAX_PAD_TRACK_BLOB_AREA
        and not _looks_like_held_cube_blob(detection)
        and (
            _looks_like_marker_blob(detection)
            or _looks_like_overhead_sign_blob(detection)
        )
    ]


def _remember_pad_bearings(memory: AgentMemory, detections: list[Any]) -> None:
    for detection in _pad_mapping_candidates(detections):
        color = normalize_color(getattr(detection, "color", None))
        if color is None:
            continue
        bearing = getattr(detection, "full_bearing_deg", getattr(detection, "angle_deg", 0.0))
        previous = memory.known_pad_bearings.get(color)
        if previous is None or detection.blob_area >= previous.get("area", 0) * 0.6:
            memory.known_pad_bearings[color] = {
                "bearing_deg": round(float(bearing), 1),
                "area": getattr(detection, "blob_area", 0),
                "bbox": getattr(detection, "bbox", None),
            }


def _known_pad_sweep_turn(memory: AgentMemory, target_color: str | None, default_turn: float) -> float:
    if not target_color:
        return default_turn
    known = memory.known_pad_bearings.get(target_color)
    if not known:
        return default_turn
    bearing = float(known.get("bearing_deg", 0.0))
    if abs(bearing) < 8.0:
        return 0.0
    return -0.45 if bearing > 0 else 0.45


def _navigation_velocity_command(
    *,
    target_kind: str,
    area: int,
    angle: float,
    arrival_area: int,
) -> tuple[float, float, float, float]:
    """Return vx, vy, wz, duration for bounded visual servoing."""
    abs_angle = abs(angle)

    if target_kind == "pad":
        if abs_angle > 8.0:
            wz = -0.3 if angle > 0 else 0.3
            vy = -0.2 if angle > 0 else 0.2
            return 0.2, vy, wz, 0.6
        wz = -angle * 0.02
        vx = 0.8 if area < arrival_area * 0.4 else 0.4
        return vx, 0.0, wz, 1.0 if vx >= 0.8 else 0.6

    if abs_angle > 16.0:
        wz = -0.4 if angle > 0 else 0.4
        return 0.0, 0.0, wz, 0.45

    if abs_angle > 8.0:
        vx = 0.35 if area < CUBE_PRECONTACT_PICK_AREA else 0.22
        wz = -angle * 0.025
        return vx, 0.0, wz, 0.45

    wz = -angle * 0.02
    if area < CUBE_PRECONTACT_PICK_AREA:
        return 0.55, 0.0, wz, 0.7
    if area < CUBE_APPROACH_PICK_AREA:
        return 0.32, 0.0, wz, 0.45
    return 0.20, 0.0, wz, 0.35


def _bbox_metrics(detection: Any) -> tuple[int, int, int, int, int, int, int, float]:
    x, y, width, height = getattr(detection, "bbox", (0, 0, 0, 0))
    cx, cy = getattr(detection, "centroid", (x + width // 2, y + height // 2))
    area = getattr(detection, "blob_area", 0)
    aspect = width / max(height, 1)
    return x, y, width, height, cx, cy, area, aspect


def _looks_like_floor_band(detection: Any) -> bool:
    """Long thin colored floor/edge strips are poor navigation anchors."""
    _, _, width, height, _, _, _, aspect = _bbox_metrics(detection)
    return (width >= 360 and height <= 120 and aspect >= 5.0) or (width >= 700 and height <= 80)


def _looks_like_marker_blob(detection: Any) -> bool:
    """표지판 마커 형태: 수직 막대/사각형, 적절한 크기."""
    x, y, width, height, cx, cy, area, aspect = _bbox_metrics(detection)
    del x, cx, cy
    # 표지판: 높이 > 너비 (수직), 적절한 크기, 바닥 띠 아님
    return (
        width >= 25
        and height >= 50
        and height >= width * 0.8
        and aspect <= 4.0
        and area <= MAX_PAD_TRACK_BLOB_AREA
        and not _looks_like_floor_band(detection)
    )


def _looks_like_overhead_sign_blob(detection: Any) -> bool:
    """표지판 형태: 화면 상단, 적절한 높이, 너무 넓지 않음."""
    _, y, width, height, _, cy, area, aspect = _bbox_metrics(detection)
    # 표지판은 대개 화면 상단 1/3에 위치, 높이 60~250px, 종횡비 0.4~3.0
    return (
        y <= 60
        and cy <= 200
        and 60 <= height <= 280
        and 40 <= width <= 400
        and 0.3 <= aspect <= 3.2
        and area <= MAX_PAD_TRACK_BLOB_AREA
    )


def _looks_like_edge_scene_blob(detection: Any) -> bool:
    x, _, width, _, cx, _, area, _ = _bbox_metrics(detection)
    touches_left_edge = x <= 3 and width >= 280
    touches_right_edge = x + width >= 1200 and width >= 280
    return area >= 20000 and (touches_left_edge or touches_right_edge or cx <= 90)


def _looks_like_wide_low_scene_patch(detection: Any) -> bool:
    _, y, width, height, _, cy, area, aspect = _bbox_metrics(detection)
    return area >= 3000 and y >= 500 and cy >= 500 and width >= 140 and height <= 180 and aspect >= 3.0


def _track_continuity_bonus(detection: Any, memory: AgentMemory, target_kind: str, target_color: str | None) -> float:
    if memory.nav_track_kind != target_kind or memory.nav_track_color != target_color:
        return 0.0
    if memory.nav_track_angle is None:
        return 0.0

    delta = abs(getattr(detection, "angle_deg", 0.0) - memory.nav_track_angle)
    return max(0.0, 35.0 - delta) * 2.2


def _pad_candidate_score(detection: Any, memory: AgentMemory, target_color: str | None) -> float:
    x, _, width, _, cx, _, area, _ = _bbox_metrics(detection)
    angle = abs(getattr(detection, "angle_deg", 0.0))

    score = max(0.0, 32.0 - angle) * 2.0
    score += min(area, MAX_PAD_TRACK_BLOB_AREA) / 1300.0
    score += _track_continuity_bonus(detection, memory, "pad", target_color)

    if _looks_like_marker_blob(detection):
        score += 35.0
    if _looks_like_floor_band(detection):
        score -= 55.0
    if x <= 3 or cx <= 35:
        score -= 15.0
    if width >= 900:
        score -= 18.0

    return score


def _cube_candidate_score(detection: Any, memory: AgentMemory, target_color: str | None) -> float:
    x, _, width, _, cx, _, area, _ = _bbox_metrics(detection)
    angle = abs(getattr(detection, "angle_deg", 0.0))

    score = max(0.0, 35.0 - angle) * 2.0
    score += min(area, MAX_CUBE_BLOB_AREA) / 1000.0
    score += _track_continuity_bonus(detection, memory, "cube", target_color)

    if _looks_like_floor_band(detection):
        score -= 60.0
    if x <= 3 or cx <= 35:
        score -= 12.0
    if width >= 700:
        score -= 30.0

    return score


def _select_target_detection(
    detections: list[Any],
    target_color: str | None,
    target_kind: str,
    memory: AgentMemory,
) -> Any | None:
    if target_kind == "pad":
        candidates = _pad_candidates(detections, target_color)
    else:
        candidates = _cube_candidates(detections, memory)
        if target_color:
            candidates = [detection for detection in candidates if detection.color == target_color]

    if not candidates:
        return None

    if target_kind == "pad":
        return max(candidates, key=lambda detection: _pad_candidate_score(detection, memory, target_color))
    return max(candidates, key=lambda detection: _cube_candidate_score(detection, memory, target_color))


def _ensure_nav_track(memory: AgentMemory, target_kind: str, target_color: str | None) -> None:
    if memory.nav_track_kind == target_kind and memory.nav_track_color == target_color:
        return
    memory.nav_track_kind = target_kind
    memory.nav_track_color = target_color
    memory.nav_track_angle = None
    memory.nav_track_lost_steps = 0


def _clear_nav_track(memory: AgentMemory) -> None:
    memory.nav_track_kind = None
    memory.nav_track_color = None
    memory.nav_track_angle = None
    memory.nav_track_lost_steps = 0


def _safe_exception_label(exc: Exception) -> str:
    message = str(exc).encode("ascii", errors="replace").decode("ascii")
    message = " ".join(message.split())
    if len(message) > 160:
        message = message[:157] + "..."
    return f"{type(exc).__name__}: {message}"


def _sdk_call_failed(result: Any) -> bool:
    if isinstance(result, dict):
        status = str(result.get("status", "")).lower()
        return bool(result.get("error")) or status in {"failed", "error", "timeout"}
    error = getattr(result, "error", None)
    status = str(getattr(result, "status", "")).lower()
    return error is not None or status in {"failed", "error", "timeout"}


def _looks_like_container_marker_blob(detection: Any) -> bool:
    """Destination anchors should be colored letter signs, not floor/hand blobs."""
    _, y, _, _, _, cy, area, _ = _bbox_metrics(detection)
    if area < 2500:
        return False
    if _looks_like_held_cube_blob(detection) or _looks_like_floor_band(detection):
        return False
    if _looks_like_overhead_sign_blob(detection):
        return True
    return _looks_like_marker_blob(detection) and y <= 260 and cy <= 340


def _is_plausible_pad_position(detection: Any) -> bool:
    """패드 표지판이 있을 법한 화면 영역인지 확인 (상단 또는 중상단)."""
    _, y, width, height, cx, cy, area, aspect = _bbox_metrics(detection)
    del area
    # 표지판은 대개 화면 상단 1/2에 위치
    if cy >= 450:
        return False
    # 너무 넓으면 벽/장애물
    if width >= 500:
        return False
    # 종횡비 극단적 제외
    if aspect >= 5.0 or aspect <= 0.15:
        return False
    # 화면 중앙 근처
    if cx <= 30 or cx >= 1170:
        return False
    # 적절한 크기
    if height <= 15 or width <= 10:
        return False
    return True


def _pad_candidates(detections: list[Any], target_color: str | None) -> list[Any]:
    candidates = [
        detection
        for detection in detections
        if (target_color is None or detection.color == target_color)
        and not _looks_like_held_cube_blob(detection)
        and _looks_like_container_marker_blob(detection)
        and getattr(detection, "blob_area", 0) <= MAX_PAD_TRACK_BLOB_AREA
        and _is_plausible_pad_position(detection)
    ]
    return sorted(candidates, key=lambda detection: detection.blob_area, reverse=True)


def _is_plausible_cube_position(detection: Any) -> bool:
    """큐브가 있을 법한 화면 영역인지 확인 (중앙, 하단, 적절한 거리)."""
    _, y, width, height, cx, cy, area, aspect = _bbox_metrics(detection)
    del height, area, aspect
    # 화면 최상단(표지판 영역)은 제외
    if y <= 30 and cy <= 120:
        return False
    # 너무 넓은 것은 벽/장애물
    if width >= 350:
        return False
    # 화면 중앙 근처 (좌우 가장자리 제외)
    if cx <= 80 or cx >= 1120:
        return False
    return True


def _cube_candidates(detections: list[Any], memory: AgentMemory) -> list[Any]:
    return [
        detection
        for detection in detections
        if detection.color not in memory.skipped_colors
        and getattr(detection, "blob_area", 0) <= MAX_CUBE_BLOB_AREA
        and getattr(detection, "blob_area", 0) >= 800  # 노이즈 제외
        and not _looks_like_floor_band(detection)
        and not _looks_like_container_marker_blob(detection)
        and not _looks_like_overhead_sign_blob(detection)
        and not _looks_like_edge_scene_blob(detection)
        and not _looks_like_wide_low_scene_patch(detection)
        and _is_plausible_cube_position(detection)
    ]


def estimate_held_color_from_detections(detections: list[Any]) -> str | None:
    held_like = [detection for detection in detections if _looks_like_held_cube_blob(detection)]
    if held_like:
        return max(held_like, key=lambda detection: detection.blob_area).color

    large = [detection for detection in detections if getattr(detection, "blob_area", 0) >= 5000]
    if large:
        return max(large, key=lambda detection: detection.blob_area).color
    return None


def _navigation_arrived(
    *,
    target_kind: str,
    area: int,
    angle_deg: float,
    moved_toward_target: bool,
    pad_direction_confirmed: bool,
    pad_forward_steps: int,
    step: int,
) -> bool:
    """Return true only when the target is close enough and centered enough."""
    arrival_area = PLACE_BLOB_AREA if target_kind == "pad" else PICK_BLOB_AREA
    centered_limit = PAD_CENTERED_DEG if target_kind == "pad" else CUBE_CENTERED_DEG
    if abs(angle_deg) > centered_limit:
        return False

    if target_kind == "cube":
        # 큐브가 충분히 가까우면 도달로 판단 (pick 가능 거리)
        if area >= arrival_area:
            return moved_toward_target or area >= CUBE_IMMEDIATE_PICK_AREA
        # 처음 몇 스텝에서 가까우면 도달
        if step <= 2 and area >= CUBE_PRECONTACT_PICK_AREA:
            return True
        # 전진하면서 접근 중이고 적절한 거리면 도달
        return moved_toward_target and step >= 3 and area >= CUBE_APPROACH_PICK_AREA

    if area < arrival_area:
        return False

    return moved_toward_target and (
        pad_forward_steps >= 3
        or (pad_direction_confirmed and pad_forward_steps >= 2)
        or step >= 10
    )


def _best_visible_cube_candidate(observation: Observation, memory: AgentMemory) -> Any | None:
    candidates = _cube_candidates(_visible_detections(observation), memory)
    if not candidates:
        return None

    if memory.active_color:
        active_target = _select_target_detection(candidates, memory.active_color, "cube", memory)
        if active_target is not None:
            return active_target

    for color in memory.priority_colors:
        prioritized_target = _select_target_detection(candidates, color, "cube", memory)
        if prioritized_target is not None:
            return prioritized_target

    return _select_target_detection(candidates, None, "cube", memory)


def choose_fast_decision(
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> AgentDecision:
    """Choose the next action locally so actions can chain without API latency."""
    if memory.delivery_limit is not None and memory.delivered_count >= memory.delivery_limit:
        return AgentDecision(next_action="stop", reason="Delivery limit reached.")

    if _result_has_failure(last_result):
        target_color = memory.held_color or memory.active_color
        return AgentDecision(
            next_action="recover",
            target_color=target_color,
            reason="Fast local policy: previous action failed, recover before retrying.",
            recovery_strategy="step_back_rescan",
        )

    if memory.held_color:
        target_color = memory.held_color
        if _failed_too_often(memory, target_color):
            return AgentDecision(
                next_action="recover",
                target_color=target_color,
                reason="Fast local policy: repeated failures while carrying cube.",
                recovery_strategy="step_back_rescan",
            )
        if memory.pad_ready:
            return AgentDecision(
                next_action="place_cube",
                target_color=target_color,
                reason="Fast local policy: matching pad navigation already succeeded.",
            )
        pad_candidates = _pad_candidates(_visible_detections(observation), target_color)
        if not pad_candidates:
            if target_color in memory.known_pad_bearings:
                return AgentDecision(
                    next_action="navigate_to_pad",
                    target_color=target_color,
                    reason="Fast local policy: matching pad bearing is mapped in memory.",
                )
            return AgentDecision(
                next_action="search_pad",
                target_color=target_color,
                reason="Fast local policy: holding a cube and no matching pad color is visible.",
            )
        return AgentDecision(
            next_action="navigate_to_pad",
            target_color=target_color,
            reason="Fast local policy: holding a cube, so navigate before placing.",
        )

    if memory.cube_ready and memory.active_color:
        return AgentDecision(
            next_action="pick_cube",
            target_color=memory.active_color,
            reason="Fast local policy: cube navigation already succeeded.",
        )

    target = _best_visible_cube_candidate(observation, memory)
    if target is None:
        memory.cube_ready = False
        search_color = _next_target_color(memory) if memory.priority_colors else None
        if _failed_too_often(memory, search_color):
            return AgentDecision(
                next_action="skip_target",
                target_color=search_color,
                reason="Fast local policy: target failed repeatedly.",
            )
        return AgentDecision(
            next_action="search_cube",
            target_color=search_color,
            reason="Fast local policy: no eligible cube color is visible.",
        )

    memory.active_color = target.color
    if _failed_too_often(memory, target.color):
        return AgentDecision(
            next_action="skip_target",
            target_color=target.color,
            reason="Fast local policy: visible target failed repeatedly.",
        )
    return AgentDecision(
        next_action="navigate_to_cube",
        target_color=target.color,
        reason="Fast local policy: cube color is visible, so navigate before picking.",
    )


async def set_head_cached(
    ctx: Any,
    memory: AgentMemory,
    *,
    yaw: float | None = None,
    pitch: float | None = None,
) -> Any:
    """Avoid an SDK action if the camera is already at the requested pose."""
    same_yaw = yaw is None or (
        memory.head_yaw is not None and abs(memory.head_yaw - yaw) <= HEAD_POSE_EPSILON
    )
    same_pitch = pitch is None or (
        memory.head_pitch is not None and abs(memory.head_pitch - pitch) <= HEAD_POSE_EPSILON
    )
    if same_yaw and same_pitch:
        return {"status": "cached"}

    result = await set_head(ctx, yaw=yaw, pitch=pitch)
    if _sdk_call_failed(result):
        return result
    if yaw is not None:
        memory.head_yaw = yaw
    if pitch is not None:
        memory.head_pitch = pitch
    return result


def parse_agent_decision(text: str) -> AgentDecision | None:
    """LLM의 JSON 응답을 parse하고 필수 schema를 검증합니다."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    next_action = data.get("next_action")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        return None

    target_color = normalize_color(data.get("target_color"))

    return AgentDecision(
        next_action=next_action,
        target_color=target_color,
        reason=str(data.get("reason", "")),
        recovery_strategy=data.get("recovery_strategy"),
    )


def build_decision_context(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """robot state를 LLM에 전달하기 좋은 compact text context로 변환합니다."""
    visible = [
        {
            "color": detection.color,
            "angle_deg": detection.angle_deg,
            "full_bearing_deg": round(getattr(detection, "full_bearing_deg", detection.angle_deg), 1),
            "blob_area": detection.blob_area,
            "bbox": detection.bbox,
        }
        for detection in observation.detections
    ]
    return {
        "task": task,
        "visible_targets": visible,
        "held_color": memory.held_color,
        "active_color": memory.active_color,
        "stage": memory.stage,
        "cube_ready": memory.cube_ready,
        "pad_ready": memory.pad_ready,
        "delivered_count": memory.delivered_count,
        "delivery_limit": memory.delivery_limit,
        "priority_colors": memory.priority_colors,
        "completed_colors": memory.completed_colors,
        "skipped_colors": memory.skipped_colors,
        "failed_attempts": memory.failed_attempts,
        "known_pad_bearings": memory.known_pad_bearings,
        "recent_outcomes": memory.recent_outcomes[-5:],
        "last_result": last_result,
        "note": observation.note,
        "signage_note": SIGNAGE_NOTE,
        "vlm_summary": observation.vlm_summary,
    }


# ---------------------------------------------------------------------------
# SUPPORT CODE: Level 2에서 허용되는 SDK wrapper
# ---------------------------------------------------------------------------
# 이 wrapper에는 scene_state, 정답 좌표, 정확한 cube/pad entity ID, go_to를
# 추가하지 마세요.

async def get_robot_status(ctx: Any, memory: AgentMemory | None = None) -> Any:
    """robot pose, motion status, neck state를 읽습니다."""
    try:
        status = await asyncio.wait_for(ctx.state("robot_status"), timeout=ROBOT_STATUS_TIMEOUT_S)
    except Exception as exc:
        error_label = _safe_exception_label(exc)
        if memory is not None:
            memory.robot_status_failures += 1
            if memory.last_robot_status is not None:
                print(f"[Status Warning] robot_status unavailable; reusing last status ({error_label})")
                return memory.last_robot_status
        print(f"[Status Warning] robot_status unavailable; using unknown fallback ({error_label})")
        return FallbackRobotStatus()

    if memory is not None:
        memory.last_robot_status = status
        memory.robot_status_failures = 0
    return status


_POV_DEBUG_FRAME_COUNT = 0


def _maybe_save_pov_frame(jpeg: bytes) -> None:
    global _POV_DEBUG_FRAME_COUNT
    if not SAVE_POV_FRAMES or POV_FRAME_LIMIT <= 0:
        return

    _POV_DEBUG_FRAME_COUNT += 1
    if _POV_DEBUG_FRAME_COUNT > POV_FRAME_LIMIT:
        return
    if (_POV_DEBUG_FRAME_COUNT - 1) % POV_FRAME_EVERY != 0:
        return

    try:
        os.makedirs(POV_FRAME_DIR, exist_ok=True)
        frame_path = os.path.join(POV_FRAME_DIR, f"pov_{_POV_DEBUG_FRAME_COUNT:04d}.jpg")
        with open(frame_path, "wb") as frame_file:
            frame_file.write(jpeg)
        if _POV_DEBUG_FRAME_COUNT <= 3 or _POV_DEBUG_FRAME_COUNT % 20 == 0:
            print(f"[POV] saved {frame_path}")
    except Exception as exc:
        print(f"[POV Warning] failed to save frame: {_safe_exception_label(exc)}")


async def get_camera_frame(ctx: Any) -> bytes:
    """POV camera frame을 가져옵니다."""
    jpeg = await ctx.get_vision("pov")
    _maybe_save_pov_frame(jpeg)
    return jpeg


def build_signage_vlm_prompt(held_color: str | None = None) -> str:
    """고정 창고 표지판을 읽기 위한 VLM prompt입니다."""
    target = ""
    if held_color in DESTINATION_SIGN_RULES:
        target = f" The robot is holding a {held_color} cube, so the target destination sign is {DESTINATION_SIGN_RULES[held_color]}."
    return (
        "Read the floating warehouse signs visible in this robot camera frame. "
        f"{SIGNAGE_NOTE} "
        "Return JSON with visible sign letters, colors, rough left/center/right positions, and confidence."
        + target
    )


async def ask_vlm_about_frame(ctx: Any, prompt: str, *, api_key: str) -> str:
    """현재 POV frame에 대해 project-allowed VLM helper로 질문합니다."""
    jpeg = await get_camera_frame(ctx)
    return ask_vlm(jpeg, prompt, api_key=api_key)


async def get_vlm_pad_direction(ctx: Any, target_color: str, *, api_key: str) -> str | None:
    sign = DESTINATION_SIGN_RULES.get(target_color, "")
    prompt = (
        "You are guiding a warehouse robot from its POV camera. "
        f"The robot is holding a {target_color} cube. Its correct destination is sign {sign}. "
        "Ignore the cube held in the robot hand. Look for the destination sign/pad in the scene. "
        "Return ONLY JSON: {\"visible\": true/false, \"direction\": \"left|center|right|unknown\", "
        "\"reason\": \"short\"}."
    )
    started = time.perf_counter()
    try:
        reply = await asyncio.wait_for(
            asyncio.to_thread(
                call_vlm_optimized,
                await get_camera_frame(ctx),
                prompt,
                api_key=api_key,
            ),
            timeout=10.5,
        )
        direction = parse_pad_hint(reply)
        print(f"VLM pad hint ({target_color}->{sign}) after {time.perf_counter() - started:.2f}s: {direction}")
        return direction
    except Exception as exc:
        print(f"VLM pad hint failed after {time.perf_counter() - started:.2f}s: {exc}")
        return None


async def perceive(ctx: Any) -> list[Any]:
    """현재 camera frame에서 Workshop 2 color-blob detector를 실행합니다."""
    jpeg = await get_camera_frame(ctx)
    return detect_color_blobs(jpeg)


async def estimate_held_color(ctx: Any, memory: AgentMemory) -> str | None:
    await set_head_cached(ctx, memory, yaw=0.0, pitch=0.02)
    await asyncio.sleep(0.1)
    return estimate_held_color_from_detections(await perceive(ctx))


async def set_head(ctx: Any, *, yaw: float | None = None, pitch: float | None = None) -> Any:
    """walking direction은 바꾸지 않고 camera 방향만 조정합니다."""
    args: dict[str, float] = {}
    if yaw is not None:
        args["yaw"] = yaw
    if pitch is not None:
        args["pitch"] = pitch
    try:
        return await ctx.invoke("set_head", args, timeout_s=HEAD_RPC_TIMEOUT_S)
    except Exception as exc:
        error = _safe_exception_label(exc)
        print(f"[Action Warning] set_head failed: {error}")
        return {"status": "failed", "error": error}


async def move_velocity(
    ctx: Any,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    duration_s: float = 1.0,
) -> Any:
    """짧은 body-frame velocity command를 보냅니다."""
    try:
        return await ctx.invoke(
            "set_velocity",
            {"vx": vx, "vy": vy, "wz": wz, "duration_s": duration_s},
            timeout_s=MOVEMENT_RPC_TIMEOUT_S,
        )
    except Exception as exc:
        error = _safe_exception_label(exc)
        print(f"[Action Warning] set_velocity failed: {error}")
        return {"status": "failed", "error": error}


async def cancel_action(ctx: Any) -> Any:
    """현재 실행 중인 runtime action을 취소합니다."""
    return await ctx.invoke("cancel", {})


async def pick_nearest_cube(ctx: Any) -> Any:
    """의도한 cube 가까이 시각적으로 이동한 뒤 nearest cube를 집습니다."""
    try:
        return await ctx.invoke(
            "pick_entity",
            {"target": {"kind": "entity", "entity_id": "cube"}},
            timeout_s=MANIPULATION_RPC_TIMEOUT_S,
        )
    except Exception as exc:
        error = _safe_exception_label(exc)
        print(f"[Action Warning] pick_entity failed: {error}")
        return {"status": "failed", "error": error}


async def place_nearest_zone(ctx: Any) -> Any:
    """matching pad 가까이 이동한 뒤 nearest zone에 내려놓습니다."""
    try:
        return await ctx.invoke("place_entity", {}, timeout_s=MANIPULATION_RPC_TIMEOUT_S)
    except Exception as exc:
        error = _safe_exception_label(exc)
        print(f"[Action Warning] place_entity failed: {error}")
        return {"status": "failed", "error": error}


def result_summary(result: Any) -> dict[str, Any]:
    """SDK result를 log에 넣기 쉬운 작은 dictionary로 변환합니다."""
    if isinstance(result, dict):
        return {
            "status": str(result.get("status")) if result.get("status") is not None else None,
            "error": result.get("error"),
        }

    error = getattr(result, "error", None)
    status = getattr(result, "status", None)
    return {
        "status": str(status) if status is not None else None,
        "error": getattr(error, "message", None) if error else None,
    }


async def scan_head(
    ctx: Any,
    *,
    yaws: tuple[float, ...] = (-0.8, 0.0, 0.8),
    pitch: float = 0.15,
    target_color: str | None = None,
) -> list[Any]:
    """간단한 scan helper입니다. target_color가 주어지면 감지 시 조기 종료합니다."""
    all_detections: list[Any] = []
    for yaw in yaws:
        await set_head(ctx, yaw=yaw, pitch=pitch)
        await asyncio.sleep(0.15) # 0.4초에서 0.15초로 대기 시간 단축
        detections = await perceive(ctx)
        found_target = False
        for detection in detections:
            scanned = ScannedDetection(
                color=detection.color,
                angle_deg=detection.angle_deg,
                blob_area=detection.blob_area,
                centroid=detection.centroid,
                bbox=detection.bbox,
                head_yaw=yaw,
                head_pitch=pitch,
            )
            all_detections.append(scanned)
            if target_color and detection.color == target_color:
                found_target = True
        if found_target:
            print(f"[Scan] Early exit: found target {target_color} at yaw {yaw}")
            break
    return all_detections


async def map_pad_by_scanning(
    ctx: Any,
    memory: AgentMemory,
    target_color: str | None,
    *,
    pitch: float,
) -> bool:
    """Map pad/sign bearings without scene_state: scan, half-turn, then small exploration moves."""
    if target_color and target_color in memory.known_pad_bearings:
        return True

    for phase in range(2):
        scanned = await scan_head(
            ctx,
            yaws=(-0.75, 0.0, 0.75),
            pitch=pitch,
            target_color=target_color,
        )
        _remember_pad_bearings(memory, scanned)
        if target_color and _select_target_detection(scanned, target_color, "pad", memory):
            await set_head_cached(ctx, memory, yaw=0.0, pitch=pitch)
            return True

        if phase == 0:
            print("[Map] Pad not found in front sweep; rotating half-turn and rescanning.")
            move_result = await move_velocity(ctx, wz=0.8, duration_s=3.2)
            if _sdk_call_failed(move_result):
                await set_head_cached(ctx, memory, yaw=0.0, pitch=pitch)
                return False

    for step in range(2):
        print(f"[Map] Pad still unseen; exploratory move {step + 1}/2.")
        move_result = await move_velocity(ctx, vx=0.25, wz=0.2, duration_s=1.0)
        if _sdk_call_failed(move_result):
            await set_head_cached(ctx, memory, yaw=0.0, pitch=pitch)
            return False
        scanned = await scan_head(
            ctx,
            yaws=(-0.75, 0.0, 0.75),
            pitch=pitch,
            target_color=target_color,
        )
        _remember_pad_bearings(memory, scanned)
        if target_color and _select_target_detection(scanned, target_color, "pad", memory):
            await set_head_cached(ctx, memory, yaw=0.0, pitch=pitch)
            return True

    await set_head_cached(ctx, memory, yaw=0.0, pitch=pitch)
    return bool(target_color and target_color in memory.known_pad_bearings)


def call_llm_optimized(
    messages: list[dict[str, Any]],
    *,
    api_key: str,
    model: str = "minimaxai/minimax-m2.7",
    timeout_s: int = 120,
    max_tokens: int = LLM_DECISION_MAX_TOKENS,
) -> str:
    """Tokamak LLM 호출 속도 최적화: temperature=0, max_tokens 제한을 통한 처리 지연 감소"""
    import requests
    response = requests.post(
        "https://api.tokamak.sh/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]
    message = choice.get("message", {})
    content = message.get("content")
    if not content:
        keys = ",".join(message.keys())
        finish = choice.get("finish_reason")
        raise ValueError(f"LLM response missing content (finish={finish}, keys={keys})")
    return content


def call_vlm_optimized(
    jpeg_bytes: bytes,
    prompt: str,
    *,
    api_key: str,
    model: str = "qwen/qwen3.6-35b-a3b",
    timeout_s: int = 10,
    max_tokens: int = 320,
) -> str:
    image_url = f"data:image/jpeg;base64,{base64.b64encode(jpeg_bytes).decode('utf-8')}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    import requests
    response = requests.post(
        "https://api.tokamak.sh/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]
    message = choice.get("message", {})
    content = message.get("content") or message.get("reasoning_content")
    if not content:
        raise ValueError(f"VLM response missing content (finish={choice.get('finish_reason')})")
    return content


def parse_pad_hint(text: str) -> str | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    blob = match.group(0) if match else ""
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        lowered = text.lower()
        if "left" in lowered:
            return "left"
        if "right" in lowered:
            return "right"
        if "center" in lowered or "straight" in lowered or "forward" in lowered:
            return "center"
        return None

    visible = data.get("visible", data.get("target_visible", data.get("found", True)))
    if visible is False or str(visible).lower() in {"false", "no", "0"}:
        return None
    direction = str(data.get("direction", data.get("position", data.get("where", "")))).lower()
    if "left" in direction:
        return "left"
    if "right" in direction:
        return "right"
    if "center" in direction or "straight" in direction or "forward" in direction:
        return "center"
    return None


def _result_has_failure(last_result: dict[str, Any] | None) -> bool:
    if not last_result:
        return False
    result = last_result.get("action_result", {})
    sdk_result = result.get("result") if isinstance(result.get("result"), dict) else {}
    status = result.get("status", sdk_result.get("status"))
    error = result.get("error", sdk_result.get("error"))
    status_text = str(status).lower() if status is not None else ""
    return (
        result.get("found") is False
        or result.get("reached") is False
        or status in ("failed", False)
        or "fail" in status_text
        or error is not None
    )


def should_consult_llm(
    fast_decision: AgentDecision,
    memory: AgentMemory,
    last_result: dict[str, Any] | None,
) -> bool:
    """Keep normal action cadence local; ask the API only for high-level recovery."""
    if fast_decision.next_action == "stop":
        return False
    if fast_decision.next_action in {"recover", "skip_target"}:
        return False
    if _result_has_failure(last_result):
        return False
    target = fast_decision.target_color or memory.held_color
    return _failed_too_often(memory, target)


# ---------------------------------------------------------------------------
async def parse_task_instructions(task: str, api_key: str) -> tuple[int | None, list[str]]:
    """자연어 지시사항을 LLM을 사용하여 파싱하고, 배달 제한 개수와 우선순위 색상 리스트를 반환합니다."""
    local_limit, local_priorities = parse_task_instructions_local(task)
    if local_limit is not None or local_priorities or task.strip() == TASK or not api_key:
        return local_limit, local_priorities

    prompt = (
        "Analyze the following natural language task instruction for a warehouse robot. "
        "Extract two pieces of information:\n"
        "1. The maximum number of cubes to deliver (if specified, e.g. 'only deliver 4' or '4개만' -> 4. If not specified or if it's the standard task of all six cubes, return null).\n"
        "2. The order of colors to prioritize (if specified, e.g. 'deliver red and blue first' or '빨간색과 파란색을 먼저' -> ['red', 'blue']. If not specified, return an empty list).\n\n"
        "Respond ONLY with a JSON object in this format:\n"
        '{"delivery_limit": int or null, "priority_colors": ["color1", "color2", ...]}\n'
        "Do not include any explanation or code blocks outside the JSON."
    )
    from menlo_runner.llm import call_llm
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": task}
    ]
    try:
        # call_llm is synchronous, run it in a thread to keep async loop responsive
        reply = await asyncio.to_thread(call_llm, messages, api_key=api_key)
        stripped = reply.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        data = json.loads(stripped)
        priorities = []
        for color in data.get("priority_colors", []):
            normalized = normalize_color(color)
            if normalized and normalized not in priorities:
                priorities.append(normalized)
        return data.get("delivery_limit"), priorities
    except Exception as e:
        print(f"Task parser error: {e}")
        return None, []


def validate_decision(decision: AgentDecision, memory: AgentMemory) -> AgentDecision:
    """LLM이 내린 고수준 결정을 메모리 상태와 대조하여 유효한지 1차 검증하고 교정합니다."""
    action = decision.next_action
    decision.target_color = normalize_color(decision.target_color)

    if action == "stop":
        limit = memory.delivery_limit
        if limit is not None and memory.delivered_count < limit:
            return AgentDecision(
                next_action="search_pad" if memory.held_color else "search_cube",
                target_color=memory.held_color,
                reason="Validation Override: delivery limit has not been reached.",
            )
        return decision

    # 1. 만약 이미 큐브를 들고 있는데 pick_cube를 하려 하면 place_cube/search_pad로 변경
    if action == "pick_cube" and memory.held_color is not None:
        print(f"[Validation Warning] Already holding {memory.held_color}. Overriding pick_cube to search_pad.")
        return AgentDecision(
            next_action="search_pad",
            target_color=memory.held_color,
            reason="Validation Override: Already holding a cube."
        )

    # 2. 큐브를 들고 있지 않은데 place_cube를 하려 하면 search_cube로 변경
    if action == "place_cube" and memory.held_color is None:
        print("[Validation Warning] Not holding any cube. Overriding place_cube to search_cube.")
        return AgentDecision(
            next_action="search_cube",
            reason="Validation Override: Not holding a cube."
        )

    if action == "pick_cube" and not memory.cube_ready:
        print("[Validation Warning] Cube navigation has not succeeded. Overriding pick_cube to navigate_to_cube.")
        return AgentDecision(
            next_action="navigate_to_cube",
            target_color=decision.target_color or memory.active_color,
            reason="Validation Override: must navigate to cube before picking."
        )

    if action == "place_cube" and not memory.pad_ready:
        print("[Validation Warning] Pad navigation has not succeeded. Overriding place_cube to navigate_to_pad.")
        return AgentDecision(
            next_action="navigate_to_pad",
            target_color=memory.held_color or decision.target_color,
            reason="Validation Override: must navigate to pad before placing."
        )

    # 3. 큐브를 들고 있는데 큐브 이동/탐색을 하거나 다른 큐브를 타겟팅하려 하면 pad 이동/탐색으로 유도
    if memory.held_color is not None and action in ("search_cube", "navigate_to_cube"):
        print(f"[Validation Warning] Holding {memory.held_color} but target is cube. Redirecting to search_pad.")
        return AgentDecision(
            next_action="search_pad",
            target_color=memory.held_color,
            reason="Validation Override: Holding a cube, must search for matching pad."
        )

    # 4. 큐브를 들고 있지 않은데 패드 이동/탐색을 하려 하면 큐브 탐색으로 변경
    if memory.held_color is None and action in ("search_pad", "navigate_to_pad"):
        print("[Validation Warning] Not holding a cube but target is pad. Redirecting to search_cube.")
        return AgentDecision(
            next_action="search_cube",
            reason="Validation Override: Not holding a cube, must search for cube."
        )

    # 5. 이미 완료(completed)되었거나 스킵(skipped)된 색상을 타겟팅하면 다른 색상으로 재유도
    target_is_completed_for_open_ended_task = (
        memory.delivery_limit is None and decision.target_color in memory.completed_colors
    )
    if target_is_completed_for_open_ended_task or decision.target_color in memory.skipped_colors:
        if action in ("search_cube", "navigate_to_cube"):
            print(f"[Validation Warning] Color {decision.target_color} is completed/skipped. Overriding target_color to None.")
            decision.target_color = None

    return decision


async def decide_next_action(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> AgentDecision:
    decision_context = build_decision_context(task, observation, memory, last_result)
    fast_decision = choose_fast_decision(observation, memory, last_result)
    if not should_consult_llm(fast_decision, memory, last_result):
        print("Fast decision: API skipped for action cadence.")
        return fast_decision

    prompt = (
        f"Task Instruction: {task}\n"
        "You are the high-level supervisor for a Level 2 autonomous vision robot.\n"
        "Follow the Task Instruction above STRICTLY and choose only the next high-level action.\n"
        "Do not output low-level set_velocity, set_head, coordinates, scene_state data, go_to targets, or entity IDs.\n"
        "- If it specifies a number of cubes (e.g., 'only deliver 4'), call stop when delivered_count reaches that number.\n"
        "- If it specifies priorities (e.g., 'red first'), target those colors first.\n"
        "- If a target fails repeatedly, use 'recover' or 'skip_target' instead of repeating forever.\n"
        "- Do NOT target colors that are in 'skipped_colors'. completed_colors is a delivery history and may contain repeats.\n"
        "Delivery rules: red->pad_B, green->pad_C, blue->pad_D, yellow->pad_E.\n"
        "You can only hold one cube at a time.\n"
        "Available next_action choices:\n"
        "  search_cube, navigate_to_cube, pick_cube, search_pad, navigate_to_pad, place_cube, recover, skip_target, stop.\n"
        "Respond ONLY with a JSON object:\n"
        '{"next_action": "action", "target_color": "color", "reason": "why"}'
    )
    decision_context["candidate_action"] = fast_decision.__dict__
    user_context = {
        "task": decision_context["task"],
        "candidate_action": decision_context["candidate_action"],
        "visible_targets": decision_context["visible_targets"][:6],
        "held_color": decision_context["held_color"],
        "stage": decision_context["stage"],
        "cube_ready": decision_context["cube_ready"],
        "pad_ready": decision_context["pad_ready"],
        "delivered_count": decision_context["delivered_count"],
        "delivery_limit": decision_context["delivery_limit"],
        "priority_colors": decision_context["priority_colors"],
        "skipped_colors": decision_context["skipped_colors"],
        "failed_attempts": decision_context["failed_attempts"],
        "recent_outcomes": decision_context["recent_outcomes"],
        "last_result": decision_context["last_result"],
    }

    api_key = os.environ.get("TOKAMAK_API_KEY", "")
    if not api_key:
        return fast_decision

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(user_context)}
    ]

    started = time.perf_counter()
    try:
        reply = await asyncio.wait_for(
            asyncio.to_thread(
                call_llm_optimized,
                messages,
                api_key=api_key,
                timeout_s=LLM_DECISION_TIMEOUT_S,
            ),
            timeout=LLM_DECISION_TIMEOUT_S + 0.5,
        )
        elapsed = time.perf_counter() - started
        decision = parse_agent_decision(reply)
        if decision:
            print(f"LLM recovery latency: {elapsed:.2f}s")
            return decision
        print(f"LLM recovery error: invalid decision JSON after {elapsed:.2f}s; using fast fallback.")
        return fast_decision
    except TimeoutError:
        elapsed = time.perf_counter() - started
        print(f"LLM recovery timeout after {elapsed:.2f}s: using fast fallback.")
        return fast_decision
    except Exception as e:
        elapsed = time.perf_counter() - started
        print(f"LLM recovery error after {elapsed:.2f}s: {e}; using fast fallback.")
        return fast_decision


# ---------------------------------------------------------------------------
# STUDENT TODO: observation, verification, memory
# ---------------------------------------------------------------------------

async def observe_world(ctx: Any, memory: AgentMemory) -> Observation:
    robot_status = await get_robot_status(ctx, memory)

    # 머리를 정면으로 고정하고 즉시 촬영 (고개 흔드는 지연을 완전히 제거)
    pitch = 0.02 if memory.held_color else 0.15
    await set_head_cached(ctx, memory, yaw=0.0, pitch=pitch)
    detections = []
    for d in await perceive(ctx):
        detections.append(ScannedDetection(
            color=d.color, angle_deg=d.angle_deg, blob_area=d.blob_area,
            centroid=d.centroid, bbox=d.bbox, head_yaw=0.0, head_pitch=pitch
        ))
    _remember_pad_bearings(memory, detections)

    return Observation(robot_status=robot_status, detections=detections)


async def verify_outcome(
    ctx: Any,
    decision: AgentDecision,
    action_result: dict[str, Any],
    memory: AgentMemory,
) -> dict[str, Any]:
    status = await get_robot_status(ctx, memory)
    held = status.robot.held_entity_ids is not None and len(status.robot.held_entity_ids) > 0
    action_result["held"] = held
    if decision.next_action == "pick_cube" and held:
        action_result["held_color_estimate"] = await estimate_held_color(ctx, memory)
    return {"decision": decision.__dict__, "action_result": action_result}


def update_memory(
    memory: AgentMemory,
    observation: Observation,
    decision: AgentDecision,
    verified: dict[str, Any],
) -> None:
    action = decision.next_action
    decision.target_color = normalize_color(decision.target_color)
    result = verified["action_result"]

    sdk_result = result.get("result") if isinstance(result.get("result"), dict) else {}
    status = result.get("status", sdk_result.get("status"))
    error = result.get("error", sdk_result.get("error"))
    status_text = str(status).lower() if status is not None else ""

    # Track failed attempts for robustness.
    success_signal = (
        result.get("reached") is True
        or result.get("found") is True
        or (action == "pick_cube" and result.get("held") is True)
        or (action == "place_cube" and result.get("held") is False and memory.held_color is not None)
        or status in ("success", True)
        or "success" in status_text
        or "complete" in status_text
        or "done" in status_text
    )
    is_success = error is None and success_signal
    is_failure = (
        result.get("reached") is False
        or result.get("found") is False
        or status in ("failed", False)
        or "fail" in status_text
        or error is not None
    )

    if action in ("navigate_to_cube", "navigate_to_pad", "pick_cube", "place_cube") and is_failure:
        color = decision.target_color or memory.held_color or "unknown"
        memory.failed_attempts[color] = memory.failed_attempts.get(color, 0) + 1
    elif is_success and decision.target_color in memory.failed_attempts:
        memory.failed_attempts[decision.target_color] = 0

    if action == "skip_target" and decision.target_color:
        memory.skipped_colors.append(decision.target_color)
        memory.active_color = None
        memory.cube_ready = False
        memory.pad_ready = False

    if action == "navigate_to_cube" and result.get("reached") is True:
        memory.active_color = decision.target_color
        memory.cube_ready = True
        memory.pad_ready = False
        memory.stage = "ready_pick"
    elif action == "navigate_to_cube" and result.get("reached") is False:
        memory.cube_ready = False
        memory.stage = "need_cube"
    elif action == "navigate_to_pad" and result.get("reached") is True:
        memory.pad_ready = True
        memory.cube_ready = False
        memory.stage = "ready_place"
    elif action == "navigate_to_pad" and result.get("reached") is False:
        memory.pad_ready = False
        memory.stage = "need_pad"
    elif action == "pick_cube" and result.get("held"):
        # 이미 들고 있었는데 또 집은 경우 방지
        if memory.held_color is not None and memory.held_color != decision.target_color:
            print(f"[State Warning] Already holding {memory.held_color}, pick_cube for {decision.target_color} ignored.")
        estimated_color = normalize_color(result.get("held_color_estimate"))
        if estimated_color and estimated_color != decision.target_color:
            print(f"[Vision] Corrected held color {decision.target_color} -> {estimated_color}")
        # 실제로 새 큐브를 들었을 때만 held_color 업데이트
        memory.held_color = estimated_color or decision.target_color
        memory.active_color = memory.held_color
        memory.cube_ready = False
        memory.pad_ready = False
        memory.stage = "need_pad"
    elif action == "place_cube" and is_success and not result.get("held") and memory.held_color:
        memory.delivered_count += 1
        memory.completed_colors.append(memory.held_color)
        memory.held_color = None
        memory.active_color = None
        memory.cube_ready = False
        memory.pad_ready = False
        memory.stage = "need_cube"
    elif action == "place_cube" and not result.get("held"):
        memory.held_color = None
        memory.active_color = None
        memory.cube_ready = False
        memory.pad_ready = False
        memory.stage = "need_cube"
    elif action == "recover":
        memory.cube_ready = False
        memory.pad_ready = False

    # Track outcomes for the recent_outcomes list (Slide 8/9/10 compliance)
    outcome = {
        "action": action,
        "target": decision.target_color,
        "success": is_success,
        "error": error
    }
    memory.recent_outcomes.append(outcome)
    if len(memory.recent_outcomes) > RECENT_OUTCOME_LIMIT:
        del memory.recent_outcomes[:-RECENT_OUTCOME_LIMIT]

    memory.logs.append({
        "observation": {
            "visible_count": len(observation.detections),
            "visible_colors": [detection.color for detection in observation.detections],
        },
        "memory": {
            "delivered_count": memory.delivered_count,
            "delivery_limit": memory.delivery_limit,
            "priority_colors": memory.priority_colors,
            "held_color": memory.held_color,
            "stage": memory.stage,
            "cube_ready": memory.cube_ready,
            "pad_ready": memory.pad_ready,
            "failed_attempts": dict(memory.failed_attempts),
            "recent_outcomes": list(memory.recent_outcomes),
        },
        "llm_decision": decision.__dict__,
        "verified": verified,
    })


# ---------------------------------------------------------------------------
# LEVEL 2 STUDENT TODO: vision-only action 구현
# ---------------------------------------------------------------------------
# Level 2에서는 go_to를 호출하면 안 됩니다. camera observation, set_head,
# set_velocity, memory, recovery behavior로 navigation을 구현하세요.

async def visual_search(
    ctx: Any,
    memory: AgentMemory,
    target_color: str | None = None,
    *,
    target_kind: str = "cube",
) -> bool:
    # 머리는 정면에 고정하고, 몸체 회전으로만 빠른 360도 스캔 수행 (고개 흔들기 대기 제거)
    pitch = 0.02 if target_kind == "pad" else 0.15
    await set_head_cached(ctx, memory, yaw=0.0, pitch=pitch)
    if target_kind == "pad" and target_color in memory.known_pad_bearings:
        return True
    for attempt in range(12):
        raw_detections = await perceive(ctx)
        detections = [
            ScannedDetection(
                color=d.color,
                angle_deg=d.angle_deg,
                blob_area=d.blob_area,
                centroid=d.centroid,
                bbox=d.bbox,
                head_yaw=0.0,
                head_pitch=pitch,
            )
            for d in raw_detections
        ]
        if target_kind == "pad":
            _remember_pad_bearings(memory, detections)
            detections = _pad_candidates(detections, target_color)
        else:
            detections = _cube_candidates(detections, memory)
        if target_color:
            if any(d.color == target_color for d in detections):
                return True
        elif len(detections) > 0:
            return True

        if target_kind == "pad" and attempt in {1, 5, 9}:
            if await map_pad_by_scanning(ctx, memory, target_color, pitch=pitch):
                return True

        print(f"Search attempt {attempt+1}: Turning body to search...")
        # 빠른 회전 속도와 짧은 동작 시간으로 대기 감소
        sweep_turn = _known_pad_sweep_turn(memory, target_color, 0.6) if target_kind == "pad" else 0.6
        move_result = await move_velocity(ctx, wz=sweep_turn, duration_s=0.6)
        if _sdk_call_failed(move_result):
            return False
    return False


async def visual_navigate_to_target(
    ctx: Any,
    memory: AgentMemory,
    target_color: str | None,
    *,
    target_kind: str,
) -> bool:
    if not target_color:
        return False

    # 루프 진입 전 고개를 한 번만 정면으로 고정하여 매 스텝마다 set_head를 호출하는 오버헤드 차단
    pitch = 0.02 if target_kind == "pad" else 0.15
    await set_head_cached(ctx, memory, yaw=0.0, pitch=pitch)
    _ensure_nav_track(memory, target_kind, target_color)

    arrival_area = PLACE_BLOB_AREA if target_kind == "pad" else PICK_BLOB_AREA
    moved_toward_target = False
    pad_direction_confirmed = False
    pad_forward_steps = 0
    last_cube_area: int | None = None
    stagnant_cube_steps = 0

    max_steps = 10 if target_kind == "pad" else 20
    for step in range(1, max_steps + 1):
        obs_list = await perceive(ctx)
        if target_kind == "pad":
            mapped_observations = [
                ScannedDetection(
                    color=d.color,
                    angle_deg=d.angle_deg,
                    blob_area=d.blob_area,
                    centroid=d.centroid,
                    bbox=d.bbox,
                    head_yaw=0.0,
                    head_pitch=pitch,
                )
                for d in obs_list
            ]
            _remember_pad_bearings(memory, mapped_observations)
        target_det = _select_target_detection(obs_list, target_color, target_kind, memory)

        if not target_det:
            memory.nav_track_lost_steps += 1
            print(f"Nav step {step}: lost {target_color}, sweeping...")
            if target_kind == "pad" and step in {1, 5, 9, 13, 17}:
                api_key = os.environ.get("TOKAMAK_API_KEY", "")
                direction = None
                if api_key and USE_VLM_PAD_HINTS:
                    direction = await get_vlm_pad_direction(ctx, target_color, api_key=api_key)
                if direction == "left":
                    pad_direction_confirmed = True
                    move_result = await move_velocity(ctx, wz=0.6, duration_s=0.8)
                    if _sdk_call_failed(move_result):
                        _clear_nav_track(memory)
                        return False
                    continue
                if direction == "right":
                    pad_direction_confirmed = True
                    move_result = await move_velocity(ctx, wz=-0.6, duration_s=0.8)
                    if _sdk_call_failed(move_result):
                        _clear_nav_track(memory)
                        return False
                    continue
                if direction == "center":
                    pad_direction_confirmed = True
                    move_result = await move_velocity(ctx, vx=0.5, duration_s=1.0)
                    if _sdk_call_failed(move_result):
                        _clear_nav_track(memory)
                        return False
                    moved_toward_target = True
                    pad_forward_steps += 1
                    if pad_forward_steps >= 3:
                        print(f"Reached {target_color} pad by repeated centered VLM guidance.")
                        _clear_nav_track(memory)
                        return True
                    continue
            sweep_dir = 0.5 if step % 2 == 0 else -0.5
            if target_kind == "pad":
                sweep_dir = _known_pad_sweep_turn(memory, target_color, sweep_dir)
            move_result = await move_velocity(ctx, wz=sweep_dir, duration_s=0.6)
            if _sdk_call_failed(move_result):
                _clear_nav_track(memory)
                return False
            continue

        area = target_det.blob_area
        angle = target_det.angle_deg
        score = (
            _pad_candidate_score(target_det, memory, target_color)
            if target_kind == "pad"
            else _cube_candidate_score(target_det, memory, target_color)
        )
        memory.nav_track_angle = angle
        memory.nav_track_lost_steps = 0
        print(
            f"Nav step {step}: {target_color} area={area} angle={angle:+.1f} score={score:.1f} "
            f"centroid={target_det.centroid} bbox={target_det.bbox}"
        )

        if _navigation_arrived(
            target_kind=target_kind,
            area=area,
            angle_deg=angle,
            moved_toward_target=moved_toward_target,
            pad_direction_confirmed=pad_direction_confirmed,
            pad_forward_steps=pad_forward_steps,
            step=step,
        ):
            print(f"Reached {target_color} target (area {area} >= {arrival_area})")
            _clear_nav_track(memory)
            return True

        if target_kind == "cube":
            if (
                last_cube_area is not None
                and moved_toward_target
                and abs(area - last_cube_area) <= CUBE_STAGNANT_AREA_DELTA
            ):
                stagnant_cube_steps += 1
            else:
                stagnant_cube_steps = 0
            last_cube_area = area

            if stagnant_cube_steps >= CUBE_STAGNANT_STEP_LIMIT:
                print(
                    f"Cube approach stagnated on {target_color}; "
                    "backing off and re-approaching from an angle."
                )
                # 후진하여 큐브를 밀지 않게 함
                await move_velocity(ctx, vx=-0.3, duration_s=1.0)
                # 좌우 중 하나로 회전하여 다른 각도에서 재접근
                turn_dir = 0.5 if step % 2 == 1 else -0.5
                await move_velocity(ctx, wz=turn_dir, duration_s=1.2)
                _clear_nav_track(memory)
                return False

        if _navigation_arrived(
            target_kind=target_kind,
            area=area,
            angle_deg=angle,
            moved_toward_target=moved_toward_target,
            pad_direction_confirmed=pad_direction_confirmed,
            pad_forward_steps=pad_forward_steps,
            step=step,
        ):
            print(f"Reached {target_color} target (area {area} >= {arrival_area})")
            _clear_nav_track(memory)
            return True

        # P-제어기를 통한 비례 조향 각속도 및 거리 기반 적응형 전진 속도/시간
        # pad 표지판은 화면 가장자리에 걸리기 쉬워 전진/횡이동/회전을 함께 써서 접근한다.
        vx, vy, wz, duration = _navigation_velocity_command(
            target_kind=target_kind,
            area=area,
            angle=angle,
            arrival_area=arrival_area,
        )
        print(
            f"Nav step {step}: Servo command {target_color} "
            f"(angle {angle:+.1f} deg, vx {vx:.2f}, vy {vy:.2f}, wz {wz:.2f})"
        )

        move_result = await move_velocity(ctx, vx=vx, vy=vy, wz=wz, duration_s=duration)
        if _sdk_call_failed(move_result):
            _clear_nav_track(memory)
            return False
        if vx > 0:
            moved_toward_target = True
            if target_kind == "pad":
                pad_forward_steps += 1

    _clear_nav_track(memory)
    return False


async def recover_motion(ctx: Any, memory: AgentMemory, reason: str | None = None) -> dict[str, Any]:
    print(f"Recovering motion. Reason: {reason}")
    if reason and "fallen" in reason.lower():
        move_result = await move_velocity(ctx, vx=0.0, wz=0.0, duration_s=2.0)
        if _sdk_call_failed(move_result):
            return {"action": "recover", "status": "failed", "error": result_summary(move_result).get("error")}
    elif memory.held_color is None:
        move_result = await move_velocity(ctx, vx=-0.35, duration_s=0.9)
        if _sdk_call_failed(move_result):
            return {"action": "recover", "status": "failed", "error": result_summary(move_result).get("error")}
        sidestep = 0.28 if memory.search_turns % 2 == 0 else -0.28
        move_result = await move_velocity(ctx, vy=sidestep, wz=0.45, duration_s=1.0)
        if _sdk_call_failed(move_result):
            return {"action": "recover", "status": "failed", "error": result_summary(move_result).get("error")}
        memory.search_turns += 1
    else:
        move_result = await move_velocity(ctx, vx=-0.4, duration_s=1.2)
        if _sdk_call_failed(move_result):
            return {"action": "recover", "status": "failed", "error": result_summary(move_result).get("error")}
        move_result = await move_velocity(ctx, wz=0.8, duration_s=1.5)
        if _sdk_call_failed(move_result):
            return {"action": "recover", "status": "failed", "error": result_summary(move_result).get("error")}
    _clear_nav_track(memory)
    return {"action": "recover", "reason": reason, "status": "stepped_back_and_rotated"}


async def execute_decision(
    ctx: Any,
    decision: AgentDecision,
    observation: Observation,
    memory: AgentMemory,
) -> dict[str, Any]:
    """검증된 LLM decision 하나를 Level 2 robot action으로 변환합니다.

    TODO:
    - go_to 없이 search/navigation을 구현하세요.
    - 의도한 cube 가까이 시각적으로 이동한 뒤 pick하세요.
    - matching pad 가까이 시각적으로 이동한 뒤 place하세요.
    - target을 잃거나 이동에 실패하면 recovery를 사용하세요.
    """
    if decision.next_action in {"search_cube", "search_pad"}:
        target_kind = "pad" if decision.next_action == "search_pad" else "cube"
        found = await visual_search(
            ctx,
            memory,
            decision.target_color,
            target_kind=target_kind,
        )
        return {"action": decision.next_action, "found": found}

    if decision.next_action in {"navigate_to_cube", "navigate_to_pad"}:
        target_kind = "pad" if decision.next_action == "navigate_to_pad" else "cube"
        reached = await visual_navigate_to_target(
            ctx,
            memory,
            decision.target_color,
            target_kind=target_kind,
        )
        return {"action": decision.next_action, "reached": reached}

    if decision.next_action == "pick_cube":
        if not memory.cube_ready:
            return {"action": "pick_cube", "status": "blocked", "error": "cube_navigation_required"}
        result = await pick_nearest_cube(ctx)
        return {"action": "pick_cube", "result": result_summary(result)}

    if decision.next_action == "place_cube":
        if not memory.pad_ready:
            return {"action": "place_cube", "status": "blocked", "error": "pad_navigation_required"}
        result = await place_nearest_zone(ctx)
        return {"action": "place_cube", "result": result_summary(result)}

    if decision.next_action == "skip_target":
        return {"action": "skip_target", "status": "success"}

    if decision.next_action == "recover":
        return await recover_motion(ctx, memory, decision.recovery_strategy)

    return {"action": decision.next_action, "status": "no_op"}


async def run_agent(ctx: Any, *, max_cycles: int = 100, task: str | None = None) -> AgentMemory:
    """얇은 observe-LLM-act loop입니다. loop만이 아니라 TODO 함수들을 수정하세요."""
    task = task or get_task_instruction()
    memory = AgentMemory()
    memory.delivery_limit = DEFAULT_DELIVERY_LIMIT

    # 1. 지시사항 동적 분석 (Slide 9/10 대응)
    api_key = os.environ.get("TOKAMAK_API_KEY", "")
    try:
        limit, priorities = await parse_task_instructions(task, api_key)
        if limit is not None:
            memory.delivery_limit = limit
        memory.priority_colors = priorities
        print(f"[Init] Parsed TASK constraints -> limit: {memory.delivery_limit}, priorities: {priorities}")
    except Exception as e:
        print(f"[Init Warning] Failed to parse task instructions: {e}")

    last_result: dict[str, Any] | None = None

    for cycle in range(1, max_cycles + 1):
        print(f"\n[Level 2] Cycle {cycle}")
        observation = await observe_world(ctx, memory)

        # 2. 결정 요청
        decision_started = time.perf_counter()
        decision = await decide_next_action(task, observation, memory, last_result)
        print(f"Decision latency: {time.perf_counter() - decision_started:.2f}s")
        print("Agent decision (Raw):", decision)

        # 3. 결정 검증 레이어 적용 (Slide 7/8 대응)
        validated_decision = validate_decision(decision, memory)
        if validated_decision != decision:
            print("Agent decision (Validated):", validated_decision)
            decision = validated_decision

        if decision.next_action == "stop":
            break

        # 4. 행동 실행 및 결과 검증/기억 업데이트
        action_started = time.perf_counter()
        action_result = await execute_decision(ctx, decision, observation, memory)
        print(f"Action latency: {time.perf_counter() - action_started:.2f}s")
        verified = await verify_outcome(ctx, decision, action_result, memory)
        update_memory(memory, observation, decision, verified)
        last_result = verified

    return memory


async def run(ctx: Any) -> None:
    task = get_task_instruction()
    print(task)
    print("Running Level 2 autonomous-vision project starter")
    memory = await run_agent(ctx, task=task)
    print("\nRun complete.")
    print(f"Delivered count: {memory.delivered_count}")
    print("Logs:")
    for item in memory.logs:
        print(item)
