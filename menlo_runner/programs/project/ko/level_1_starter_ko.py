from __future__ import annotations

"""Menlo AI 로봇 분류 챌린지용 Level 1 프로젝트 시작 파일입니다.

이 파일은 완성된 해답이 아니라 시작 파일입니다.

지원 코드 섹션은 반복해서 작성할 필요가 없는 작은 래퍼와 자료 구조를 제공합니다.
필요하면 읽고 수정할 수 있지만, 대부분의 팀은 지원 코드를 크게 바꾸지 않는 편이 좋습니다.
학생 TODO 섹션은 팀이 수정하고, 개선하고, test하고, presentation에서 설명해야 하는 부분입니다.

Level 1 규칙: scene_state와 정확한 entity ID는 사용할 수 없습니다. Coordinate go_to는
학생 시스템이 관찰로 추정하거나 기록한 좌표에만 사용할 수 있습니다.
"""

import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from menlo_runner.completion import CompletionConfig, CompletionTracker
from menlo_runner.llm import ask_vlm
from menlo_runner.perception import annotate_detections, decode_jpeg, detect_color_blobs
from menlo_runner.scene import delivered_cube_ids, held_cube_info

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# 지원 코드: 공통 과제 정의와 필수 LLM 결정 형식
# ---------------------------------------------------------------------------
# 과제 문장은 고정합니다. 목표는 cube 색상 순서와 시작 위치가 달라져도
# 소스 코드 변경 없이 처리하는 하나의 agent를 만드는 것입니다.
TASK = "Find and sort cubes from the source area into their matching destination pads."

# 고정 표지판 정보는 사용할 수 있습니다. 단, 이를 정확한 coordinate나 entity ID로
# 바꾸지 말고 관찰을 해석하는 데만 사용하세요.
DESTINATION_SIGN_RULES = {
    "red": "B",
    "green": "C",
    "blue": "D",
    "yellow": "E",
}
SIGNAGE_NOTE = (
    "A는 conveyor/cube source area이며 destination이 아닙니다. "
    "Destination sign은 B red, C green, D blue, E yellow입니다."
)

COLOR_ORDER = ("red", "green", "blue", "yellow")
CUBE_NAV_PITCH = float(os.environ.get("MENLO_LEVEL1_CUBE_PITCH", "0.34"))
PAD_NAV_PITCH = float(os.environ.get("MENLO_LEVEL1_PAD_PITCH", "0.35"))
CLOSE_LOOK_PITCH = float(os.environ.get("MENLO_LEVEL1_CLOSE_PITCH", "0.50"))
ROBOT_STATUS_TIMEOUT_S = float(os.environ.get("MENLO_LEVEL1_STATE_TIMEOUT_S", "6"))
VISION_TIMEOUT_S = float(os.environ.get("MENLO_LEVEL1_VISION_TIMEOUT_S", "8"))
HEAD_TIMEOUT_S = float(os.environ.get("MENLO_LEVEL1_HEAD_TIMEOUT_S", "6"))
GO_TO_TIMEOUT_S = float(os.environ.get("MENLO_LEVEL1_GOTO_TIMEOUT_S", "55"))
PICK_TIMEOUT_S = float(os.environ.get("MENLO_LEVEL1_PICK_TIMEOUT_S", "35"))
PLACE_TIMEOUT_S = float(os.environ.get("MENLO_LEVEL1_PLACE_TIMEOUT_S", "35"))
RUNTIME_READY_TIMEOUT_S = float(os.environ.get("MENLO_LEVEL1_READY_TIMEOUT_S", "40"))
LLM_ADVICE_TIMEOUT_S = float(os.environ.get("MENLO_LEVEL1_LLM_TIMEOUT_S", "3.5"))
LLM_ADVICE_EVERY_N_CYCLES = int(os.environ.get("MENLO_LEVEL1_LLM_EVERY_N", "25"))
SIM_SPEED = float(os.environ.get("MENLO_SIM_SPEED", "1.0"))
MAX_PAD_NAV_STEP_M = float(os.environ.get("MENLO_LEVEL1_MAX_PAD_NAV_STEP_M", "2.1"))
SOURCE_REVISIT_RADIUS_M = float(os.environ.get("MENLO_LEVEL1_SOURCE_REVISIT_RADIUS_M", "0.85"))
SOURCE_PICK_COOLDOWN_CYCLES = int(os.environ.get("MENLO_LEVEL1_SOURCE_PICK_COOLDOWN", "2"))
NAV_REACHED_TOLERANCE_M = float(os.environ.get("MENLO_LEVEL1_NAV_REACHED_TOLERANCE_M", "0.72"))
PAD_REACHED_TOLERANCE_M = float(os.environ.get("MENLO_LEVEL1_PAD_REACHED_TOLERANCE_M", "0.45"))
PAD_MIN_LETTER_SCORE = float(os.environ.get("MENLO_LEVEL1_PAD_MIN_LETTER_SCORE", "0.10"))
PAD_MIN_WOOD_SCORE = float(os.environ.get("MENLO_LEVEL1_PAD_MIN_WOOD_SCORE", "0.025"))
PAD_MIN_GREEN_LETTER_SCORE = float(os.environ.get("MENLO_LEVEL1_PAD_MIN_GREEN_LETTER_SCORE", "0.055"))
PAD_MIN_GREEN_WOOD_SCORE = float(os.environ.get("MENLO_LEVEL1_PAD_MIN_GREEN_WOOD_SCORE", "0.015"))
PAD_STRICT_FEATURES = os.environ.get("MENLO_LEVEL1_PAD_STRICT_FEATURES", "1").lower() not in {"0", "false", "no"}
DEBUG_PAD_REJECTS = os.environ.get("MENLO_LEVEL1_DEBUG_PAD_REJECTS", "1").lower() not in {"0", "false", "no"}
SAVE_POV_FRAMES = os.environ.get("MENLO_LEVEL1_SAVE_POV", "1").lower() not in {"0", "false", "no"}
POV_FRAME_DIR = os.environ.get("MENLO_LEVEL1_POV_DIR", os.path.join("run_logs", "pov_frames"))
POV_MAX_FRAMES = int(os.environ.get("MENLO_LEVEL1_POV_MAX_FRAMES", "240"))
SOURCE_NUDGE_DURATION_S = float(os.environ.get("MENLO_LEVEL1_SOURCE_NUDGE_DURATION_S", "0.85"))
NO_PLACE_RADIUS_M = float(os.environ.get("MENLO_LEVEL1_NO_PLACE_RADIUS_M", "0.95"))
HAZARD_RADIUS_M = float(os.environ.get("MENLO_LEVEL1_HAZARD_RADIUS_M", "0.85"))
CONFIRMED_PLACE_RADIUS_M = float(os.environ.get("MENLO_LEVEL1_CONFIRMED_PLACE_RADIUS_M", "0.90"))
DROP_APPROACH_RADIUS_M = float(os.environ.get("MENLO_LEVEL1_DROP_APPROACH_RADIUS_M", "1.20"))
UNCONFIRMED_PAD_MAX_STEP_M = float(os.environ.get("MENLO_LEVEL1_UNCONFIRMED_PAD_MAX_STEP_M", "1.35"))
SEMANTIC_MAP_LOG = os.environ.get("MENLO_LEVEL1_MAP_LOG", os.path.join("run_logs", "level1_worldmap.jsonl"))
PAD_SCAN_YAWS = tuple(math.radians(deg) for deg in (-50, 0, 50))

# LLM은 아래 set에서 상위 단계 행동을 선택해야 합니다. 원시 속도 명령을
# 직접 출력하지 말고, 결정적 코드가 결정을 robot 행동으로 변환해야 합니다.
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


@dataclass
class AgentDecision:
    """LLM이 반환하고 코드가 검증한 상위 단계 결정입니다."""

    next_action: str
    target_color: str | None = None
    reason: str = ""
    recovery_strategy: str | None = None


@dataclass
class AgentMemory:
    """observe-decide-act cycle 사이에 agent가 유지하는 상태입니다.

    간단하게 시작한 뒤, 팀 전략에 필요한 field를 추가하세요. 예: target history,
    failed location, scan result, confidence score, held-object estimate 등.
    """

    delivered_count: int = 0
    baseline_delivered_count: int | None = None
    held_color: str | None = None
    active_color: str | None = None
    stage: str = "need_cube"
    cube_ready: bool = False
    pad_ready: bool = False
    active_mode: str = "deliver"
    active_target_xy: tuple[float, float] | None = None
    active_target_kind: str | None = None
    target_pad_xy: tuple[float, float] | None = None
    known_pad_xy: dict[str, tuple[float, float]] = field(default_factory=dict)
    confirmed_pad_xy: dict[str, tuple[float, float]] = field(default_factory=dict)
    rejected_pad_xy: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    blocked_pad_xy: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    pad_certificates: dict[str, dict[str, Any]] = field(default_factory=dict)
    candidate_pad_xy: dict[str, tuple[float, float]] = field(default_factory=dict)
    landmark_rays: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    no_place_zones: list[dict[str, Any]] = field(default_factory=list)
    hazard_zones: list[dict[str, Any]] = field(default_factory=list)
    safe_scan_hubs: list[tuple[float, float]] = field(default_factory=list)
    semantic_map_events: list[dict[str, Any]] = field(default_factory=list)
    known_source_xy: tuple[float, float] | None = None
    priority_colors: list[str] = field(default_factory=list)
    blocked_colors: dict[str, int] = field(default_factory=dict)
    source_pick_cooldown: int = 0
    diagnostic_frames: list[str] = field(default_factory=list)
    search_turns: int = 0
    failed_attempts: dict[str, int] = field(default_factory=dict)
    completed_colors: list[str] = field(default_factory=list)
    skipped_colors: list[str] = field(default_factory=list)
    recent_outcomes: list[dict[str, Any]] = field(default_factory=list)
    llm_notes: list[str] = field(default_factory=list)
    last_robot_status: Any | None = None
    robot_status_failures: int = 0
    last_robot_xy: tuple[float, float] | None = None
    last_robot_z: float | None = None
    last_scan_attempts: int = 0
    last_scan_failures: int = 0
    consecutive_rpc_failed_scans: int = 0
    fallen_detected: bool = False
    cycle_index: int = 0
    logs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Observation:
    """LLM과 실행 코드에 전달할 간결한 관찰입니다."""

    robot_status: Any
    detections: list[Any]
    note: str = ""
    vlm_summary: str = ""


@dataclass(frozen=True)
class ScannedDetection:
    """해당 camera frame을 얻을 때 사용한 head pose가 함께 기록된 color detection입니다.

    이 구조는 특정 strategy에 묶이지 않도록 의도적으로 중립적입니다. Level 1 팀은 coordinate estimate에 full bearing을 사용할 수 있고, Level 2 팀은 closed-loop visual centering에 사용할 수 있습니다. 필요하면 confidence, target type, depth field를 추가하세요.
    """

    color: str
    angle_deg: float
    blob_area: int
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    head_yaw: float
    head_pitch: float
    letter_score: float = 0.0
    wood_score: float = 0.0
    feature_ready: bool = False

    @property
    def full_bearing_deg(self) -> float:
        """대략적인 body-relative bearing입니다. Image angle에 head yaw를 더합니다."""
        return self.angle_deg + math.degrees(self.head_yaw)


def parse_agent_decision(text: str) -> AgentDecision | None:
    """필수 structured LLM JSON output을 parse하고 validate합니다."""
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

    target_color = data.get("target_color")
    if target_color is not None and not isinstance(target_color, str):
        return None

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
    """Robot state를 LLM에 전달하기 좋은 간결한 text context로 변환합니다.

    VLM을 명시적으로 사용하는 경우가 아니라면 raw image는 이 text context에 넣지 마세요. LLM은 다음 high-level step을 고를 만큼의 정보만 받고, low-level control과 safety는 code가 처리해야 합니다.
    """
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
        "delivered_count": memory.delivered_count,
        "completed_colors": memory.completed_colors,
        "skipped_colors": memory.skipped_colors,
        "failed_attempts": memory.failed_attempts,
        "last_result": last_result,
        "note": observation.note,
        "signage_note": SIGNAGE_NOTE,
        "vlm_summary": observation.vlm_summary,
    }


# ---------------------------------------------------------------------------
# 지원 코드: project 규칙에 맞는 SDK wrapper
# ---------------------------------------------------------------------------
# 이 래퍼들은 프로젝트 규칙에 맞는 input을 노출합니다. 아래 progress helper는
# completion과 robot이 cube를 들고 있는지 추적할 수 있도록 허용됩니다.
# Ground-truth coordinate, 정확한 target ID, global asset map은 추가하지 마세요.

async def get_robot_status(ctx: Any) -> Any:
    """Robot pose, motion status, neck state를 읽습니다."""
    return await asyncio.wait_for(ctx.state("robot_status"), timeout=ROBOT_STATUS_TIMEOUT_S)


async def get_camera_frame(ctx: Any) -> bytes:
    """POV camera frame을 가져옵니다."""
    return await asyncio.wait_for(ctx.get_vision("pov"), timeout=VISION_TIMEOUT_S)


async def get_delivered_count(ctx: Any) -> int:
    """공통 workshop progress helper로 delivered cube 수를 셉니다."""
    return len(await delivered_cube_ids(ctx))


async def get_held_cube_info(ctx: Any) -> dict[str, str] | None:
    """Robot이 cube를 들고 있으면 현재 held cube color만 반환합니다."""
    held = await held_cube_info(ctx)
    return {"color": held[1]} if held else None


def build_signage_vlm_prompt(held_color: str | None = None) -> str:
    """고정 warehouse signage를 읽기 위한 strategy-neutral prompt를 만듭니다."""
    target = ""
    if held_color in DESTINATION_SIGN_RULES:
        target = f" Robot이 {held_color} cube를 들고 있으므로 target destination sign은 {DESTINATION_SIGN_RULES[held_color]}입니다."
    return (
        "이 robot camera frame에 보이는 warehouse sign을 읽으세요. "
        f"{SIGNAGE_NOTE} "
        "보이는 sign letter, color, 대략적인 left/center/right 위치, confidence를 JSON으로 반환하세요."
        + target
    )


async def ask_vlm_about_frame(ctx: Any, prompt: str, *, api_key: str) -> str:
    """Project에서 허용되는 VLM helper로 현재 POV frame에 대해 질문합니다."""
    jpeg = await get_camera_frame(ctx)
    return ask_vlm(jpeg, prompt, api_key=api_key)


async def perceive(ctx: Any) -> list[Any]:
    """현재 camera frame에서 Workshop 2 color-blob detector를 실행합니다."""
    jpeg = await get_camera_frame(ctx)
    return detect_color_blobs(jpeg)


def _cv2_np() -> tuple[Any, Any]:
    import cv2
    import numpy as np

    return cv2, np


def _visual_features_for_detection(image: Any, detection: Any) -> tuple[float, float]:
    """Return (white-letter score, wood-pallet score) for a color blob bbox."""
    cv2, np = _cv2_np()
    height, width = image.shape[:2]
    x, y, bbox_width, bbox_height = getattr(detection, "bbox", (0, 0, 0, 0))
    if bbox_width <= 0 or bbox_height <= 0:
        return 0.0, 0.0

    pad = max(3, int(min(bbox_width, bbox_height) * 0.08))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + bbox_width + pad)
    y2 = min(height, y + bbox_height + pad)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0, 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Colored signs have white letters inside the colored rectangle. Cubes on
    # the conveyor usually do not, even if they are visually large.
    white_mask = (hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 155)
    letter_score = float(np.count_nonzero(white_mask)) / float(white_mask.size)

    below_y1 = min(height, y + bbox_height + 2)
    below_y2 = min(height, y + bbox_height + max(28, int(bbox_height * 2.2)))
    below_x1 = max(0, x - int(bbox_width * 0.9))
    below_x2 = min(width, x + bbox_width + int(bbox_width * 0.9))
    if below_y2 <= below_y1 or below_x2 <= below_x1:
        return letter_score, 0.0
    below = image[below_y1:below_y2, below_x1:below_x2]
    hsv_below = cv2.cvtColor(below, cv2.COLOR_BGR2HSV)
    hue = hsv_below[:, :, 0]
    sat = hsv_below[:, :, 1]
    val = hsv_below[:, :, 2]
    wood_mask = (hue >= 5) & (hue <= 32) & (sat > 35) & (val > 35) & (val < 235)
    wood_score = float(np.count_nonzero(wood_mask)) / float(wood_mask.size)
    return letter_score, wood_score


_POV_FRAME_COUNT = 0


def _safe_label(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)[:90]


def _save_pov_frame(jpeg: bytes, label: str, detections: list[Any] | None = None) -> str | None:
    """Save an annotated robot POV frame for live visual forensics."""
    global _POV_FRAME_COUNT
    if not SAVE_POV_FRAMES or _POV_FRAME_COUNT >= POV_MAX_FRAMES:
        return None
    try:
        Path(POV_FRAME_DIR).mkdir(parents=True, exist_ok=True)
        frame_bytes = annotate_detections(jpeg, detections) if detections is not None else jpeg
        path = Path(POV_FRAME_DIR) / f"level1_{_POV_FRAME_COUNT:04d}_{_safe_label(label)}.jpg"
        path.write_bytes(frame_bytes)
        _POV_FRAME_COUNT += 1
        return str(path)
    except Exception as exc:
        print(f"[Forensics Warning] could not save POV frame: {type(exc).__name__}: {exc}")
        return None


async def set_head(ctx: Any, *, yaw: float | None = None, pitch: float | None = None) -> Any:
    """Walking direction을 바꾸지 않고 camera 방향을 조정합니다."""
    args: dict[str, float] = {}
    if yaw is not None:
        args["yaw"] = yaw
    if pitch is not None:
        args["pitch"] = pitch
    return await ctx.invoke("set_head", args, timeout_s=HEAD_TIMEOUT_S)


async def move_velocity(
    ctx: Any,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    duration_s: float = 1.0,
) -> Any:
    """짧은 body-frame velocity command를 보낸 뒤 멈춥니다."""
    return await ctx.invoke(
        "set_velocity",
        {"vx": vx, "vy": vy, "wz": wz, "duration_s": duration_s},
        timeout_s=max(5.0, duration_s + 4.0),
    )


async def cancel_action(ctx: Any) -> Any:
    """현재 실행 중인 runtime action을 취소합니다."""
    return await ctx.invoke("cancel", {})


async def pick_nearest_cube(ctx: Any) -> Any:
    """Code가 robot을 시각적으로 충분히 위치시킨 뒤 nearest cube를 집습니다."""
    return await ctx.invoke(
        "pick_entity",
        {"target": {"kind": "entity", "entity_id": "cube"}},
        timeout_s=PICK_TIMEOUT_S,
    )


async def place_nearest_zone(ctx: Any) -> Any:
    """Matching pad에 도달한 뒤 nearest zone에 place합니다."""
    return await ctx.invoke("place_entity", {}, timeout_s=PLACE_TIMEOUT_S)


def result_summary(result: Any) -> dict[str, Any]:
    """SDK result를 log하기 쉬운 작은 dictionary로 변환합니다."""
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
    memory: AgentMemory | None = None,
) -> list[Any]:
    """간단한 scan helper입니다. 더 나은 search 전략으로 교체할 수 있습니다."""
    all_detections: list[Any] = []
    failures = 0
    for yaw in yaws:
        try:
            await set_head(ctx, yaw=yaw, pitch=pitch)
            await asyncio.sleep(0.35)
            jpeg = await get_camera_frame(ctx)
            image = decode_jpeg(jpeg)
            frame_detections = detect_color_blobs(jpeg)
            frame_path = _save_pov_frame(
                jpeg,
                f"cycle{memory.cycle_index if memory else 0:04d}_yaw{yaw:+.2f}_pitch{pitch:+.2f}",
                frame_detections,
            )
            if frame_path and memory is not None:
                memory.diagnostic_frames.append(frame_path)
                memory.diagnostic_frames = memory.diagnostic_frames[-24:]
        except Exception as exc:
            failures += 1
            print(f"[Vision Warning] scan skipped yaw={yaw:.2f}: {type(exc).__name__}: {exc}")
            continue
        for detection in frame_detections:
            letter_score, wood_score = _visual_features_for_detection(image, detection)
            all_detections.append(
                ScannedDetection(
                    color=detection.color,
                    angle_deg=detection.angle_deg,
                    blob_area=detection.blob_area,
                    centroid=detection.centroid,
                    bbox=detection.bbox,
                    head_yaw=yaw,
                    head_pitch=pitch,
                    letter_score=letter_score,
                    wood_score=wood_score,
                    feature_ready=True,
                )
            )
    if memory is not None:
        memory.last_scan_attempts = len(yaws)
        memory.last_scan_failures = failures
        if yaws and failures >= len(yaws):
            memory.consecutive_rpc_failed_scans += 1
        else:
            memory.consecutive_rpc_failed_scans = 0
    return all_detections


async def close_visual_observation(ctx: Any, memory: AgentMemory, *, yaw: float = 0.0, pitch: float = CLOSE_LOOK_PITCH) -> Observation:
    """Collect one downward close-look frame for guarded pick/place checks."""
    try:
        await set_head(ctx, yaw=yaw, pitch=pitch)
        await asyncio.sleep(0.25)
    except Exception:
        pass
    status = await get_robot_status_safe(ctx, memory)
    try:
        jpeg = await get_camera_frame(ctx)
        image = decode_jpeg(jpeg)
        raw_detections = detect_color_blobs(jpeg)
    except Exception:
        return Observation(robot_status=status, detections=[], note="close_visual_failed")
    detections: list[Any] = []
    for detection in raw_detections:
        letter_score, wood_score = _visual_features_for_detection(image, detection)
        detections.append(
            ScannedDetection(
                color=detection.color,
                angle_deg=detection.angle_deg,
                blob_area=detection.blob_area,
                centroid=detection.centroid,
                bbox=detection.bbox,
                head_yaw=yaw,
                head_pitch=pitch,
                letter_score=letter_score,
                wood_score=wood_score,
                feature_ready=True,
            )
        )
    frame_path = _save_pov_frame(jpeg, f"cycle{memory.cycle_index:04d}_close_yaw{yaw:+.2f}_pitch{pitch:+.2f}", detections)
    if frame_path:
        memory.diagnostic_frames.append(frame_path)
        memory.diagnostic_frames = memory.diagnostic_frames[-24:]
    observation = Observation(robot_status=status, detections=detections, note="close_visual")
    _remember_pad_estimates(observation, memory)
    return observation


# ---------------------------------------------------------------------------
# 학생 TODO: LLM decision 함수
# ---------------------------------------------------------------------------
async def decide_next_action(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> AgentDecision:
    """고속 local policy를 기본으로 쓰고, LLM은 짧은 advisory로만 참여합니다."""
    local_decision = _choose_local_decision(observation, memory, last_result)
    advice = await _ask_llm_advice(task, observation, memory, last_result, local_decision)
    if advice is None:
        return local_decision
    return _validate_decision(advice, local_decision, memory)


def _observation_scan_plan(memory: AgentMemory) -> tuple[tuple[float, ...], float] | None:
    """Choose the cheapest scan needed for the current behavior-tree state."""
    if memory.stage == "ready_place":
        return None
    if memory.held_color:
        if memory.held_color in memory.known_pad_xy:
            return None
        return (-0.65, 0.0, 0.65), PAD_NAV_PITCH
    if memory.stage == "ready_pick":
        return (0.0,), CLOSE_LOOK_PITCH
    if memory.known_source_xy is not None and memory.source_pick_cooldown <= 0:
        return None
    return (-0.55, 0.0, 0.55), CUBE_NAV_PITCH


# ---------------------------------------------------------------------------
# 학생 TODO: observation, execution, verification, memory
# ---------------------------------------------------------------------------
async def observe_world(ctx: Any, memory: AgentMemory) -> Observation:
    """LLM과 실행 코드를 위해 현재 관찰을 수집합니다.

    scene_state 없이 robot_status와 camera observation만 사용합니다.
    """
    robot_status = await get_robot_status_safe(ctx, memory)
    scan_plan = _observation_scan_plan(memory)
    if scan_plan is None:
        detections = []
        memory.last_scan_attempts = 0
        memory.last_scan_failures = 0
    else:
        yaws, pitch = scan_plan
        detections = await scan_head(ctx, yaws=yaws, pitch=pitch, memory=memory)

    observation = Observation(robot_status=robot_status, detections=detections)
    _remember_pad_estimates(observation, memory)
    if not memory.priority_colors and len(memory.known_pad_xy) >= 2 and not _status_unavailable(robot_status):
        rx, ry, _ = _robot_xy_yaw(robot_status)
        ranked = sorted(
            memory.known_pad_xy,
            key=lambda color: math.hypot(memory.known_pad_xy[color][0] - rx, memory.known_pad_xy[color][1] - ry),
        )
        memory.priority_colors = ranked[:2]
    observation.note = (
        f"stage={memory.stage}; held={memory.held_color}; "
        f"known_pads={memory.known_pad_xy}; priority={memory.priority_colors}"
    )
    return observation


async def verify_outcome(ctx: Any, decision: AgentDecision, action_result: dict[str, Any]) -> dict[str, Any]:
    """마지막 action이 성공한 것처럼 보이는지 확인합니다.

    TODO:
    - 중요한 action 뒤에는 다시 observe하세요.
    - robot_status, camera evidence, SDK result status를 확인하세요.
    - 다음 LLM call이 recovery에 사용할 수 있는 정보를 반환하세요.
    """
    try:
        held = await asyncio.wait_for(get_held_cube_info(ctx), timeout=ROBOT_STATUS_TIMEOUT_S)
    except Exception as exc:
        held = None
        action_result = {**action_result, "verify_warning": f"held check failed: {type(exc).__name__}: {exc}"}
    try:
        delivered_count = await asyncio.wait_for(get_delivered_count(ctx), timeout=ROBOT_STATUS_TIMEOUT_S)
    except Exception as exc:
        delivered_count = None
        action_result = {**action_result, "progress_warning": f"delivered count failed: {type(exc).__name__}: {exc}"}
    try:
        status = await get_robot_status(ctx)
        robot_x, robot_y, robot_yaw = _robot_xy_yaw(status)
        robot_xy = (robot_x, robot_y)
        robot_z = _robot_z(status)
    except Exception as exc:
        robot_xy = None
        robot_yaw = None
        robot_z = None
        action_result = {**action_result, "status_warning": f"robot status after action failed: {type(exc).__name__}: {exc}"}
    pov_frame = None
    if SAVE_POV_FRAMES and action_result.get("action") in {"navigate_to_cube", "pick_cube", "navigate_to_pad", "place_cube", "recover"}:
        try:
            jpeg = await get_camera_frame(ctx)
            detections = detect_color_blobs(jpeg)
            pov_frame = _save_pov_frame(jpeg, f"cycle_post_{action_result.get('action')}", detections)
        except Exception as exc:
            action_result = {**action_result, "pov_warning": f"post-action POV failed: {type(exc).__name__}: {exc}"}

    verified = {
        "decision": decision.__dict__,
        "action_result": action_result,
        "robot_xy": robot_xy,
        "robot_yaw": robot_yaw,
        "robot_z": robot_z,
        "pov_frame": pov_frame,
        "held_cube": held,
        "held_color": held["color"] if held else None,
    }
    if delivered_count is not None:
        verified["delivered_count"] = delivered_count
    return verified


def update_memory(
    memory: AgentMemory,
    observation: Observation,
    decision: AgentDecision,
    verified: dict[str, Any],
) -> None:
    """각 cycle 뒤 지속 상태를 update합니다.

    Pick 뒤 실제 held color를 기준으로 target을 보정하고, 실패는 bounded recovery로 넘깁니다.
    """
    previous_delivered_count = memory.delivered_count
    verified_delivered_count = int(verified.get("delivered_count", previous_delivered_count))
    if "delivered_count" in verified:
        if memory.baseline_delivered_count is None:
            memory.baseline_delivered_count = verified_delivered_count
        memory.delivered_count = max(memory.delivered_count, verified_delivered_count)
    if verified.get("robot_xy") is not None:
        memory.last_robot_xy = tuple(verified["robot_xy"])
    if verified.get("robot_z") is not None:
        memory.last_robot_z = float(verified["robot_z"])
    if memory.last_robot_xy is not None:
        _safe_scan_hub(memory, memory.last_robot_xy)
    memory.blocked_colors = {
        color: remaining - 1
        for color, remaining in memory.blocked_colors.items()
        if remaining > 1
    }
    memory.blocked_pad_xy = {
        color: [
            {**block, "ttl": int(block.get("ttl", 1)) - 1}
            for block in blocks
            if int(block.get("ttl", 1)) > 1
        ]
        for color, blocks in memory.blocked_pad_xy.items()
        if any(int(block.get("ttl", 1)) > 1 for block in blocks)
    }
    memory.held_color = verified.get("held_color")
    action_result = verified.get("action_result", {})
    action = action_result.get("action")
    ok = action_result.get("ok") is True
    error_text = _action_error_text(action_result).lower()
    if "fallen" in error_text:
        memory.fallen_detected = True
        memory.stage = "stopped"
        if memory.last_robot_xy is not None:
            _add_hazard_zone(memory, memory.last_robot_xy, reason="fallen", color=decision.target_color)

    if action == "navigate_to_cube":
        memory.cube_ready = ok
        memory.stage = "ready_pick" if ok else "need_cube"
        if not ok and decision.target_color:
            memory.failed_attempts[decision.target_color] = memory.failed_attempts.get(decision.target_color, 0) + 1
            memory.active_target_xy = None
            memory.active_color = None

    elif action == "pick_cube":
        if ok and memory.held_color:
            memory.source_pick_cooldown = 0
            if memory.last_robot_xy is not None:
                memory.known_source_xy = memory.last_robot_xy
                _add_no_place_zone(memory, memory.last_robot_xy, reason="source_conveyor_A", radius=0.45)
            elif memory.active_target_xy is not None:
                memory.known_source_xy = memory.active_target_xy
                _add_no_place_zone(memory, memory.active_target_xy, reason="source_conveyor_A", radius=0.45)
            memory.blocked_colors.pop(memory.held_color, None)
            memory.active_color = memory.held_color
            memory.cube_ready = False
            memory.pad_ready = False
            memory.stage = "need_pad"
        elif memory.held_color:
            memory.stage = "need_pad"
        else:
            memory.stage = "need_cube"
            memory.cube_ready = False
            if decision.target_color:
                memory.failed_attempts[decision.target_color] = memory.failed_attempts.get(decision.target_color, 0) + 1
                memory.blocked_colors[decision.target_color] = 4
            if memory.active_target_kind == "source":
                memory.source_pick_cooldown = SOURCE_PICK_COOLDOWN_CYCLES
            memory.active_color = None
            memory.active_target_xy = None
            memory.active_target_kind = None

    elif action == "navigate_to_pad":
        partial = bool(action_result.get("partial"))
        memory.pad_ready = ok and not partial
        memory.stage = "need_pad" if partial else ("ready_place" if ok else "need_pad")
        if not ok and decision.target_color:
            memory.failed_attempts[decision.target_color] = memory.failed_attempts.get(decision.target_color, 0) + 1
            preflight_block = action_result.get("reason") in {
                "no_place_zone",
                "hazard_zone",
                "source_or_conveyor_zone",
                "probable_A_source_sign",
                "held_or_floor_reflection",
                "temporary_blocked_pad_xy",
            }
            if decision.target_color not in memory.confirmed_pad_xy and memory.target_pad_xy is not None and not preflight_block:
                memory.rejected_pad_xy.setdefault(decision.target_color, []).append(memory.target_pad_xy)
                memory.rejected_pad_xy[decision.target_color] = memory.rejected_pad_xy[decision.target_color][-4:]
            elif preflight_block:
                blocked_xy = action_result.get("target_xy") or memory.target_pad_xy
                if blocked_xy is not None:
                    _add_temporary_blocked_pad_xy(
                        memory,
                        decision.target_color,
                        tuple(blocked_xy),
                        reason=action_result.get("reason", "preflight_block"),
                    )
            if decision.target_color not in memory.confirmed_pad_xy:
                memory.known_pad_xy.pop(decision.target_color, None)
                memory.candidate_pad_xy.pop(decision.target_color, None)
                memory.target_pad_xy = None

    elif action == "place_cube":
        scored_delivery = verified_delivered_count > previous_delivered_count
        placed = ok and action_result.get("was_holding") and memory.held_color is None and scored_delivery
        target_color = decision.target_color
        if placed:
            if target_color and memory.target_pad_xy is not None:
                memory.confirmed_pad_xy[target_color] = memory.target_pad_xy
                memory.known_pad_xy[target_color] = memory.target_pad_xy
                memory.pad_certificates.pop(target_color, None)
                _map_event(memory, "confirmed_drop_zone", color=target_color, xy=memory.target_pad_xy)
            memory.active_color = None
            memory.active_target_xy = None
            memory.target_pad_xy = None
            memory.cube_ready = False
            memory.pad_ready = False
            memory.stage = "need_cube"
        else:
            if target_color and target_color not in memory.confirmed_pad_xy:
                if memory.target_pad_xy is not None:
                    memory.rejected_pad_xy.setdefault(target_color, []).append(memory.target_pad_xy)
                    memory.rejected_pad_xy[target_color] = memory.rejected_pad_xy[target_color][-4:]
                    if action_result.get("attempted_place", True):
                        _add_no_place_zone(memory, memory.target_pad_xy, reason="place_without_score", color=target_color, radius=NO_PLACE_RADIUS_M)
                memory.known_pad_xy.pop(target_color, None)
                memory.candidate_pad_xy.pop(target_color, None)
                memory.pad_certificates.pop(target_color, None)
            memory.target_pad_xy = None
            memory.stage = "need_pad" if memory.held_color else "need_cube"
            memory.pad_ready = False

    elif action == "recover":
        memory.stage = "need_pad" if memory.held_color else "need_cube"
        memory.cube_ready = False
        memory.pad_ready = False

    outcome = {
        "action": action,
        "target": decision.target_color,
        "success": ok,
        "error": _action_error_text(action_result),
        "failure_mode": _failure_mode(action_result),
        "duration_s": action_result.get("duration_s"),
        "delivered_count": memory.delivered_count,
        "scored_delta": _scored_delta(memory),
        "stage": memory.stage,
    }
    memory.recent_outcomes.append(outcome)
    memory.recent_outcomes = memory.recent_outcomes[-10:]

    memory.logs.append({
        "observation": {
            "visible_count": len(observation.detections),
            "top_detections": [_detection_log(detection) for detection in observation.detections[:8]],
            "note": observation.note,
            "delivered_count": memory.delivered_count,
            "baseline_delivered_count": memory.baseline_delivered_count,
            "held_color": memory.held_color,
        },
        "decision": decision.__dict__,
        "memory": {
            "stage": memory.stage,
            "active_color": memory.active_color,
            "active_target_xy": memory.active_target_xy,
            "target_pad_xy": memory.target_pad_xy,
            "known_pad_xy": memory.known_pad_xy,
            "candidate_pad_xy": memory.candidate_pad_xy,
            "confirmed_pad_xy": memory.confirmed_pad_xy,
            "rejected_pad_xy": memory.rejected_pad_xy,
            "blocked_pad_xy": memory.blocked_pad_xy,
            "pad_certificates": memory.pad_certificates,
            "no_place_zones": memory.no_place_zones[-6:],
            "hazard_zones": memory.hazard_zones[-6:],
            "safe_scan_hubs": memory.safe_scan_hubs[-6:],
            "priority_colors": memory.priority_colors,
            "blocked_colors": memory.blocked_colors,
            "source_pick_cooldown": memory.source_pick_cooldown,
            "failed_attempts": memory.failed_attempts,
            "last_robot_xy": memory.last_robot_xy,
            "last_robot_z": memory.last_robot_z,
            "diagnostic_frames": memory.diagnostic_frames[-6:],
            "scan_failures": (memory.last_scan_failures, memory.last_scan_attempts),
            "fallen_detected": memory.fallen_detected,
            "recent_outcomes": memory.recent_outcomes[-5:],
            "llm_notes": memory.llm_notes[-3:],
        },
        "verified": verified,
    })


# ---------------------------------------------------------------------------
# Fast Level 1 implementation helpers
# ---------------------------------------------------------------------------

def _sdk_ok(result: Any) -> bool:
    if isinstance(result, dict):
        return str(result.get("status", "")).lower() == "done"
    return str(getattr(result, "status", "")).lower() == "done"


def _scored_delta(memory: AgentMemory) -> int:
    baseline = memory.baseline_delivered_count
    if baseline is None:
        baseline = memory.delivered_count
    return max(0, memory.delivered_count - baseline)


def _action_error_text(action_result: dict[str, Any]) -> str:
    parts = [
        action_result.get("error"),
        action_result.get("reason"),
        action_result.get("status"),
    ]
    result = action_result.get("result")
    if isinstance(result, dict):
        parts.extend([result.get("error"), result.get("status")])
    return " ".join(str(part) for part in parts if part is not None)


def _failure_mode(action_result: dict[str, Any]) -> str | None:
    if action_result.get("ok") is True:
        return None
    action = str(action_result.get("action", ""))
    text = _action_error_text(action_result).lower()
    if "fallen" in text:
        return "fallen"
    if "timeout" in text or "rpc" in text or "no reply" in text or "unavailable" in text:
        return "rpc_ready"
    if action == "pick_cube":
        return "source_pick"
    if action in {"navigate_to_cube", "navigate_to_pad", "recover"}:
        return "navigation"
    if action == "place_cube":
        return "place_verify"
    if action == "search_pad":
        return "wrong_pad"
    return "unknown"


def _detection_log(detection: Any) -> dict[str, Any]:
    return {
        "color": getattr(detection, "color", None),
        "area": getattr(detection, "blob_area", None),
        "centroid": getattr(detection, "centroid", None),
        "bbox": getattr(detection, "bbox", None),
        "bearing": round(float(getattr(detection, "full_bearing_deg", getattr(detection, "angle_deg", 0.0))), 2),
        "letter": round(float(getattr(detection, "letter_score", 0.0)), 4),
        "wood": round(float(getattr(detection, "wood_score", 0.0)), 4),
        "feature_ready": bool(getattr(detection, "feature_ready", False)),
    }


def _result_error(result: Any) -> str | None:
    error = getattr(result, "error", None)
    return getattr(error, "message", None) if error else None


def _fallback_status() -> Any:
    pose = SimpleNamespace(position=(0.0, 0.0, 0.0), yaw_deg=0.0)
    robot = SimpleNamespace(pose=pose, held_entity_ids=[])
    return SimpleNamespace(robot=robot, unavailable=True)


async def get_robot_status_safe(ctx: Any, memory: AgentMemory) -> Any:
    try:
        status = await get_robot_status(ctx)
    except Exception as exc:
        memory.robot_status_failures += 1
        print(f"[State Warning] robot_status unavailable: {type(exc).__name__}: {exc}")
        return memory.last_robot_status or _fallback_status()
    memory.last_robot_status = status
    return status


async def wait_for_runtime_ready(ctx: Any, memory: AgentMemory) -> bool:
    """Wait until robot_status and POV camera are both responsive before scoring."""
    deadline = time.monotonic() + RUNTIME_READY_TIMEOUT_S
    attempt = 0
    last_error = ""
    while time.monotonic() < deadline:
        attempt += 1
        try:
            status = await get_robot_status(ctx)
            jpeg = await get_camera_frame(ctx)
            if jpeg:
                memory.last_robot_status = status
                memory.last_robot_xy = _robot_xy_yaw(status)[:2]
                memory.last_robot_z = _robot_z(status)
                print(
                    "[Warmup] runtime ready "
                    f"attempt={attempt} robot_xy={memory.last_robot_xy}"
                )
                return True
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[Warmup] waiting for runtime attempt={attempt}: {last_error}")
            await asyncio.sleep(1.0)
    print(f"[Warmup Warning] runtime not fully ready: {last_error}")
    return False


def _robot_xy_yaw(status: Any) -> tuple[float, float, float]:
    pose = status.robot.pose
    position = pose.position
    return float(position[0]), float(position[1]), float(getattr(pose, "yaw_deg", 0.0))


def _robot_z(status: Any) -> float | None:
    try:
        return float(status.robot.pose.position[2])
    except Exception:
        return None


def _status_unavailable(status: Any) -> bool:
    return bool(getattr(status, "unavailable", False))


def _bearing_world_deg(status: Any, detection: Any) -> float:
    _, _, yaw_deg = _robot_xy_yaw(status)
    bearing = getattr(detection, "full_bearing_deg", getattr(detection, "angle_deg", 0.0))
    return yaw_deg + float(bearing)


def _xy_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _map_event(memory: AgentMemory, kind: str, **data: Any) -> None:
    event = {"kind": kind, "cycle": memory.cycle_index, "time": round(time.time(), 3), **data}
    memory.semantic_map_events.append(event)
    memory.semantic_map_events = memory.semantic_map_events[-80:]
    if not SEMANTIC_MAP_LOG:
        return
    try:
        path = Path(SEMANTIC_MAP_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def _llm_backend_config(env: dict[str, str] | None = None) -> tuple[str | None, str | None, str]:
    env = env or os.environ
    tokamak_api_key = env.get("TOKAMAK_API_KEY")
    if tokamak_api_key:
        return "https://api.tokamak.sh/v1/chat/completions", tokamak_api_key, env.get("TOKAMAK_MODEL", "qwen/qwen3.6-35b-a3b")
    openrouter_api_key = env.get("OPENROUTER_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN")
    return "https://openrouter.ai/api/v1/chat/completions", openrouter_api_key, env.get("OPENROUTER_MODEL", "openrouter/free")


def _add_zone(memory: AgentMemory, zones: list[dict[str, Any]], xy: tuple[float, float], *, reason: str, radius: float, color: str | None = None) -> None:
    for zone in zones:
        if _xy_distance(tuple(zone["xy"]), xy) < max(radius, float(zone.get("radius", radius))) * 0.55:
            zone["radius"] = max(float(zone.get("radius", radius)), radius)
            zone["reason"] = reason
            zone["color"] = color
            return
    zones.append({"xy": xy, "radius": radius, "reason": reason, "color": color})
    del zones[:-18]
    _map_event(memory, "zone_added", xy=xy, radius=radius, reason=reason, color=color)


def _add_no_place_zone(memory: AgentMemory, xy: tuple[float, float], *, reason: str, color: str | None = None, radius: float = NO_PLACE_RADIUS_M) -> None:
    _add_zone(memory, memory.no_place_zones, xy, reason=reason, radius=radius, color=color)


def _add_hazard_zone(memory: AgentMemory, xy: tuple[float, float], *, reason: str, color: str | None = None, radius: float = HAZARD_RADIUS_M) -> None:
    _add_zone(memory, memory.hazard_zones, xy, reason=reason, radius=radius, color=color)


def _xy_in_zones(xy: tuple[float, float], zones: list[dict[str, Any]], *, radius_scale: float = 1.0) -> bool:
    return any(_xy_distance(tuple(zone["xy"]), xy) <= float(zone.get("radius", 0.0)) * radius_scale for zone in zones)


def _safe_scan_hub(memory: AgentMemory, xy: tuple[float, float]) -> None:
    if any(_xy_distance(xy, hub) < 0.55 for hub in memory.safe_scan_hubs):
        return
    memory.safe_scan_hubs.append(xy)
    memory.safe_scan_hubs = memory.safe_scan_hubs[-8:]
    _map_event(memory, "safe_scan_hub", xy=xy)


def _is_rejected_pad_xy(memory: AgentMemory, color: str, xy: tuple[float, float]) -> bool:
    return any(_xy_distance(xy, bad_xy) < 1.0 for bad_xy in memory.rejected_pad_xy.get(color, []))


def _add_temporary_blocked_pad_xy(
    memory: AgentMemory,
    color: str,
    xy: tuple[float, float],
    *,
    reason: str,
    radius: float = 0.72,
    ttl: int = 5,
) -> None:
    blocks = memory.blocked_pad_xy.setdefault(color, [])
    if any(_xy_distance(tuple(block["xy"]), xy) < radius for block in blocks):
        for block in blocks:
            if _xy_distance(tuple(block["xy"]), xy) < radius:
                block["ttl"] = max(int(block.get("ttl", 1)), ttl)
                block["reason"] = reason
        return
    blocks.append({"xy": xy, "radius": radius, "ttl": ttl, "reason": reason})
    memory.blocked_pad_xy[color] = blocks[-8:]
    _map_event(memory, "temporary_blocked_pad", color=color, xy=xy, reason=reason, ttl=ttl)


def _is_temporarily_blocked_pad_xy(memory: AgentMemory, color: str, xy: tuple[float, float]) -> bool:
    return any(
        _xy_distance(xy, tuple(block["xy"])) <= float(block.get("radius", 0.72))
        for block in memory.blocked_pad_xy.get(color, [])
    )


def _is_plausible_pad_xy(xy: tuple[float, float]) -> bool:
    x, y = xy
    return -3.8 <= x <= 4.8 and -4.2 <= y <= 2.2


def _is_unusable_pad_xy(memory: AgentMemory, color: str, xy: tuple[float, float]) -> bool:
    if (
        not _is_plausible_pad_xy(xy)
        or _is_rejected_pad_xy(memory, color, xy)
        or _is_temporarily_blocked_pad_xy(memory, color, xy)
        or _xy_in_zones(xy, memory.no_place_zones)
        or _xy_in_zones(xy, memory.hazard_zones)
    ):
        return True
    for other_color, confirmed_xy in memory.confirmed_pad_xy.items():
        if other_color != color and _xy_distance(xy, confirmed_xy) < 0.95:
            return True
    # C/green can sit immediately beside the A conveyor in the scored scenes.
    # B/D/E are farther away, so reject their source-near color blobs more
    # aggressively while still allowing the real C pallet.
    if memory.known_source_xy is not None:
        source_distance = _xy_distance(xy, memory.known_source_xy)
        if color == "green":
            min_source_distance = 0.25
        elif color == "blue":
            min_source_distance = 0.45
        else:
            min_source_distance = 1.15
        if source_distance < min_source_distance:
            return True
    return False


def _navigation_block_reason(memory: AgentMemory, xy: tuple[float, float], *, purpose: str, color: str | None = None) -> str | None:
    if not _is_plausible_pad_xy(xy) and purpose == "pad":
        return "implausible_pad_xy"
    if color and _is_temporarily_blocked_pad_xy(memory, color, xy):
        return "temporary_blocked_pad_xy"
    if color and _is_rejected_pad_xy(memory, color, xy):
        return "rejected_pad_xy"
    if _xy_in_zones(xy, memory.hazard_zones):
        return "hazard_zone"
    if purpose in {"pad", "place"} and _xy_in_zones(xy, memory.no_place_zones):
        return "no_place_zone"
    if purpose in {"pad", "place"} and memory.known_source_xy is not None and _xy_distance(xy, memory.known_source_xy) < 0.65:
        return "source_or_conveyor_zone"
    return None


def _is_probable_source_a_sign(memory: AgentMemory, status: Any, detection: Any, target_xy: tuple[float, float]) -> bool:
    if getattr(detection, "color", None) != "green" or memory.known_source_xy is None or _status_unavailable(status):
        return False
    robot_xy = _robot_xy_yaw(status)[:2]
    source_xy = memory.known_source_xy
    if _xy_distance(robot_xy, source_xy) > 2.7:
        return False
    bearing = float(getattr(detection, "full_bearing_deg", getattr(detection, "angle_deg", 0.0)))
    _, y, _, _, _, cy, area, _ = _bbox_metrics(detection)
    # From the A/source belt, the real C pallet is usually off to the left,
    # while the green A source sign sits ahead/right on the conveyor.
    if bearing > -8.0 and y >= 220 and cy >= 260 and 2500 <= area <= 18000:
        return True
    # Live forensics: while farming the source, green false positives on the
    # A/conveyor side and rack reflections repeatedly projected to the right
    # half of the robot. Treat those as source-side no-place hints, not C.
    if (
        bearing > 35.0
        and _xy_distance(target_xy, source_xy) < 2.55
        and 180 <= y <= 360
        and 220 <= cy <= 430
        and 2000 <= area <= 22000
    ):
        return True
    if _xy_distance(target_xy, source_xy) < 0.95:
        return True
    return False


def _is_probable_held_or_floor_reflection(
    memory: AgentMemory,
    status: Any,
    detection: Any,
    target_xy: tuple[float, float] | None = None,
) -> bool:
    if getattr(detection, "color", None) != memory.held_color or memory.held_color is None:
        return False
    if _status_unavailable(status):
        return False
    _, y, width, height, _, cy, area, _ = _bbox_metrics(detection)
    letter_score = float(getattr(detection, "letter_score", 0.0))
    wood_score = float(getattr(detection, "wood_score", 0.0))

    # A carried cube and glossy floor reflections stay low in the POV and can
    # accidentally inherit wood/letter scores from the background crop.
    if y >= 390 or cy >= 455:
        return True
    if cy >= 380 and area >= 9000 and letter_score < 0.04:
        return True

    if target_xy is not None:
        robot_xy = _robot_xy_yaw(status)[:2]
        if cy >= 330 and _xy_distance(robot_xy, target_xy) < 0.45 and letter_score < 0.12:
            return True

    # Thin same-color strips near the lower half are usually gripper/cube
    # artifacts, not a pallet face.
    if cy >= 330 and (height < 42 or width / max(height, 1) > 4.0) and wood_score < 0.20:
        return True
    return False


def _robot_height_safe(robot_z: float | None) -> bool:
    return robot_z is None or robot_z >= 0.42


def _pad_visual_evidence(detection: Any, color: str) -> bool:
    if getattr(detection, "color", None) != color:
        return False
    if _looks_like_landmark_only(detection) or _bbox_is_edge_clipped(detection):
        return False
    if _looks_like_pad_candidate(detection):
        return True
    if not getattr(detection, "feature_ready", False):
        return False
    letter_score = float(getattr(detection, "letter_score", 0.0))
    wood_score = float(getattr(detection, "wood_score", 0.0))
    if color == "green":
        return letter_score >= PAD_MIN_GREEN_LETTER_SCORE and wood_score >= PAD_MIN_GREEN_WOOD_SCORE
    return letter_score >= PAD_MIN_LETTER_SCORE and wood_score >= PAD_MIN_WOOD_SCORE


def _make_pallet_certificate(
    observation: Observation,
    memory: AgentMemory,
    color: str | None,
    *,
    target_xy: tuple[float, float] | None = None,
) -> dict[str, Any] | None:
    if color not in COLOR_ORDER or target_xy is None or _status_unavailable(observation.robot_status):
        return None
    robot_xy = _robot_xy_yaw(observation.robot_status)[:2]
    robot_z = _robot_z(observation.robot_status)
    if not _robot_height_safe(robot_z):
        _add_hazard_zone(memory, robot_xy, reason="unsafe_robot_height", color=color)
        return None
    block_reason = _navigation_block_reason(memory, target_xy, purpose="place", color=color)
    if block_reason:
        return None
    distance_to_target = _xy_distance(robot_xy, target_xy)
    if color in memory.confirmed_pad_xy and distance_to_target <= CONFIRMED_PLACE_RADIUS_M:
        cert = {
            "color": color,
            "target_xy": target_xy,
            "robot_xy": robot_xy,
            "distance_m": round(distance_to_target, 3),
            "source": "confirmed_drop_zone",
        }
        memory.pad_certificates[color] = cert
        return cert
    if distance_to_target > DROP_APPROACH_RADIUS_M:
        return None
    evidence = [
        detection
        for detection in observation.detections
        if _pad_visual_evidence(detection, color)
        and not _is_probable_held_or_floor_reflection(memory, observation.robot_status, detection, target_xy)
    ]
    if not evidence:
        return None
    best = max(
        evidence,
        key=lambda detection: (
            float(getattr(detection, "wood_score", 0.0)) * 2.0 + float(getattr(detection, "letter_score", 0.0)),
            getattr(detection, "blob_area", 0),
        ),
    )
    cert = {
        "color": color,
        "target_xy": target_xy,
        "robot_xy": robot_xy,
        "distance_m": round(distance_to_target, 3),
        "bbox": getattr(best, "bbox", None),
        "letter": round(float(getattr(best, "letter_score", 0.0)), 3),
        "wood": round(float(getattr(best, "wood_score", 0.0)), 3),
        "source": "near_pallet_visual",
    }
    memory.pad_certificates[color] = cert
    _map_event(memory, "pallet_certificate", color=color, certificate=cert)
    return cert


def _project_detection_xy(
    status: Any,
    detection: Any,
    *,
    target_kind: str,
) -> tuple[float, float]:
    rx, ry, _ = _robot_xy_yaw(status)
    distance = _estimate_detection_distance(detection, target_kind=target_kind)
    bearing_rad = math.radians(_bearing_world_deg(status, detection))
    return (
        rx + math.cos(bearing_rad) * distance,
        ry + math.sin(bearing_rad) * distance,
    )


def _bounded_step_xy(
    current_xy: tuple[float, float],
    target_xy: tuple[float, float],
    *,
    max_step_m: float,
) -> tuple[tuple[float, float], bool]:
    distance = _xy_distance(current_xy, target_xy)
    if distance <= max_step_m:
        return target_xy, False
    ratio = max_step_m / max(distance, 1e-6)
    return (
        (
            current_xy[0] + (target_xy[0] - current_xy[0]) * ratio,
            current_xy[1] + (target_xy[1] - current_xy[1]) * ratio,
        ),
        True,
    )


def _pad_search_motion(memory: AgentMemory) -> tuple[dict[str, float], str]:
    """Robot-vacuum style pad search after wide head scans fail."""
    phase = memory.search_turns % 4
    memory.search_turns += 1
    if phase == 0:
        return {"wz": 0.6, "duration_s": 4.6}, "rotate_about_180"
    if phase == 1:
        return {"vx": -0.12, "vy": 0.12, "wz": 0.20, "duration_s": 1.0}, "back_left_rescan"
    if phase == 2:
        return {"wz": -0.6, "duration_s": 3.2}, "rotate_back"
    return {"vx": 0.12, "vy": -0.08, "wz": -0.18, "duration_s": 1.0}, "probe_forward_right"


def _bbox_metrics(detection: Any) -> tuple[int, int, int, int, int, int, int, float]:
    x, y, width, height = getattr(detection, "bbox", (0, 0, 0, 0))
    cx, cy = getattr(detection, "centroid", (x + width // 2, y + height // 2))
    area = int(getattr(detection, "blob_area", 0))
    aspect = width / max(height, 1)
    return x, y, width, height, cx, cy, area, aspect


def _bbox_is_edge_clipped(detection: Any, *, frame_width: int = 1280, frame_height: int = 720, margin: int = 3) -> bool:
    x, y, width, height, _, _, _, _ = _bbox_metrics(detection)
    return x <= margin or y <= margin or x + width >= frame_width - margin or y + height >= frame_height - margin


def _looks_like_landmark_only(detection: Any) -> bool:
    x, y, width, height, _, cy, area, _ = _bbox_metrics(detection)
    letter_score = float(getattr(detection, "letter_score", 0.0))
    wood_score = float(getattr(detection, "wood_score", 0.0))
    if getattr(detection, "color", None) == "blue" and _looks_like_blue_pallet_candidate(detection):
        return False
    if _looks_like_blue_d_landmark(detection):
        return True
    if _bbox_is_edge_clipped(detection) and area >= 2400:
        return True
    return area >= 9000 and width >= 110 and height >= 85 and cy >= 160 and letter_score >= 0.04 and wood_score < 0.06


def _estimate_detection_distance(detection: Any, *, target_kind: str) -> float:
    _, _, width, height, _, cy, area, _ = _bbox_metrics(detection)
    if area <= 0:
        return 2.0
    if target_kind == "pad":
        wood_score = float(getattr(detection, "wood_score", 0.0))
        far_blue_pallet_ray = (
            getattr(detection, "color", None) == "blue"
            and getattr(detection, "feature_ready", False)
            and wood_score >= 0.45
            and cy < 170
            and area < 4500
        )
        base = 230.0 / math.sqrt(max(area, 1))
        height_hint = 95.0 / max(height, 1)
        distance = 0.65 * base + 0.35 * height_hint
        if cy < 180:
            distance += 0.5
        if far_blue_pallet_ray:
            distance += 1.15
        # Project the observed sign/pallet center first. navigate_to_pad applies
        # the physical standoff later; subtracting here as well made live runs
        # stop at A/conveyor instead of reaching the C pallet approach zone.
        if (
            getattr(detection, "color", None) == "blue"
            and area >= 9000
            and width >= 100
            and height >= 85
            and cy >= 180
        ):
            # D is often seen from the A conveyor behind the big source sign.
            # The pallet is farther along the bearing than the blue sign crop
            # estimate suggests; the generic standoff otherwise drops at A.
            distance += 0.45
        if area > 12000:
            distance -= 0.15
        if far_blue_pallet_ray:
            return max(2.75, min(3.95, distance))
        return max(0.40, min(2.35, distance))

    # Live testing showed the first implementation projected conveyor cubes
    # several meters past the source belt. For Level 1, arriving within pick
    # range matters more than perfectly centering a far-away visual blob.
    base = 62.0 / math.sqrt(max(area, 1))
    height_hint = 38.0 / max(height, 1)
    distance = 0.7 * base + 0.3 * height_hint
    if cy < 260:
        distance *= 0.75
    if width < 45 or height < 45:
        distance *= 1.15
    distance -= 0.28
    return max(0.45, min(1.35, distance))


def _looks_like_cube_candidate(detection: Any) -> bool:
    _, y, width, height, _, cy, area, aspect = _bbox_metrics(detection)
    if area < 850 or area > 32000:
        return False
    if width > 220 or height > 260:
        return False
    if not (0.45 <= aspect <= 2.2):
        return False
    if cy < 145:
        return False
    if y > 620 and width > 160:
        return False
    return True


def _looks_like_close_source_cube(detection: Any) -> bool:
    """Conveyor cube is close enough to try generic pick without slow go_to."""
    _, y, width, height, _, cy, area, aspect = _bbox_metrics(detection)
    if area < 480 or area > 32000:
        return False
    if width > 230 or height > 270:
        return False
    if not (0.35 <= aspect <= 2.4):
        return False
    return cy >= 560 or y >= 500


def _source_pick_visible(observation: Observation) -> bool:
    return any(_looks_like_close_source_cube(detection) for detection in observation.detections)


def _looks_like_pad_shape_candidate(detection: Any) -> bool:
    x, y, width, height, cx, cy, area, aspect = _bbox_metrics(detection)
    if area < 1800 or area > 42000:
        return False
    if width < 45 or height < 45:
        return False
    if not (0.35 <= aspect <= 2.2):
        return False
    if width > 260 or height > 260:
        return False
    overhead_or_sign = cy <= 410 and y <= 430
    side_close_sign = cy <= 500 and (x <= 180 or cx >= 1080)
    if not (overhead_or_sign or side_close_sign):
        return False
    return True


def _looks_like_blue_d_landmark(detection: Any) -> bool:
    x, y, width, height, _, cy, area, _ = _bbox_metrics(detection)
    return (
        getattr(detection, "color", None) == "blue"
        and area >= 9000
        and width >= 100
        and height >= 85
        and y >= 80
        and 180 <= cy <= 560
    )


def _looks_like_blue_pallet_candidate(detection: Any) -> bool:
    if getattr(detection, "color", None) != "blue" or not getattr(detection, "feature_ready", False):
        return False
    x, y, width, height, _, cy, area, aspect = _bbox_metrics(detection)
    wood_score = float(getattr(detection, "wood_score", 0.0))
    side_clipped = x <= 3 or x + width >= 1280 - 3
    vertical_clipped = y <= 3 or y + height >= 720 - 3
    if _bbox_is_edge_clipped(detection):
        high_conf_side_view = (
            side_clipped
            and not vertical_clipped
            and 1800 <= area <= 5500
            and width >= 55
            and height >= 45
            and 12 <= y <= 220
            and cy <= 230
            and wood_score >= 0.70
        )
        if not high_conf_side_view:
            return False
    return (
        1600 <= area <= 6500
        and width >= 60
        and height >= 45
        and 0.75 <= aspect <= 2.1
        and 12 <= y <= 220
        and cy <= 220
        and wood_score >= 0.32
    )


def _looks_like_pad_candidate(detection: Any) -> bool:
    if not _looks_like_pad_shape_candidate(detection):
        return False
    if _looks_like_landmark_only(detection):
        return False
    x, y, width, height, _, cy, area, _ = _bbox_metrics(detection)
    if getattr(detection, "feature_ready", False):
        letter_score = float(getattr(detection, "letter_score", 0.0))
        wood_score = float(getattr(detection, "wood_score", 0.0))
        if detection.color == "blue":
            if _looks_like_blue_pallet_candidate(detection):
                return True
            if _looks_like_blue_d_landmark(detection):
                return False
        if _bbox_is_edge_clipped(detection):
            return False
        min_letter = PAD_MIN_GREEN_LETTER_SCORE if detection.color == "green" else PAD_MIN_LETTER_SCORE
        min_wood = PAD_MIN_GREEN_WOOD_SCORE if detection.color == "green" else PAD_MIN_WOOD_SCORE
        if PAD_STRICT_FEATURES:
            if letter_score < min_letter or wood_score < min_wood:
                return False
        elif letter_score < min_letter and wood_score < min_wood:
            return False
    return True


def _score_cube_candidate(detection: Any, memory: AgentMemory) -> float:
    _, _, _, _, _, cy, area, _ = _bbox_metrics(detection)
    score = min(area, 28000) / 900.0
    score += max(0, cy - 180) / 25.0
    score -= abs(float(getattr(detection, "full_bearing_deg", getattr(detection, "angle_deg", 0.0)))) / 3.0
    if memory.priority_colors and detection.color in memory.priority_colors:
        score += 5.0
    score -= memory.failed_attempts.get(detection.color, 0) * 8.0
    score -= memory.blocked_colors.get(detection.color, 0) * 14.0
    return score


def _score_pad_candidate(detection: Any, memory: AgentMemory, target_color: str) -> float:
    _, _, _, _, _, cy, area, aspect = _bbox_metrics(detection)
    bearing = float(getattr(detection, "full_bearing_deg", getattr(detection, "angle_deg", 0.0)))
    score = min(area, 35000) / 1300.0
    score -= abs(bearing) / 4.0
    score -= abs(1.0 - aspect) * 4.0
    score += float(getattr(detection, "letter_score", 0.0)) * 90.0
    score += float(getattr(detection, "wood_score", 0.0)) * 45.0
    if 80 <= cy <= 420:
        score += 8.0
    if target_color in memory.known_pad_xy:
        score += 5.0
    if target_color == "green" and memory.known_source_xy is not None:
        if -55.0 <= bearing <= 15.0:
            score += 24.0
        elif bearing > 35.0:
            score -= 48.0
    return score


def _cube_candidates(observation: Observation, memory: AgentMemory) -> list[Any]:
    candidates = [
        detection
        for detection in observation.detections
        if _looks_like_cube_candidate(detection)
        and detection.color not in memory.skipped_colors
    ]
    return sorted(candidates, key=lambda detection: _score_cube_candidate(detection, memory), reverse=True)


def _pad_candidates(observation: Observation, memory: AgentMemory, target_color: str) -> list[Any]:
    candidates = [
        detection
        for detection in observation.detections
        if detection.color == target_color and _looks_like_pad_candidate(detection)
    ]
    return sorted(candidates, key=lambda detection: _score_pad_candidate(detection, memory, target_color), reverse=True)


def _landmark_seek_candidates(observation: Observation, memory: AgentMemory, target_color: str) -> list[Any]:
    status = observation.robot_status
    if target_color not in COLOR_ORDER or _status_unavailable(status):
        return []
    candidates: list[Any] = []
    for detection in observation.detections:
        if getattr(detection, "color", None) != target_color or not getattr(detection, "feature_ready", False):
            continue
        x, y, width, height, _, cy, area, _ = _bbox_metrics(detection)
        letter_score = float(getattr(detection, "letter_score", 0.0))
        if letter_score < 0.075:
            continue
        if not (2400 <= area <= 42000 and 45 <= width <= 190 and 55 <= height <= 270 and cy <= 430):
            continue
        target_xy = _project_detection_xy(status, detection, target_kind="pad")
        if _is_probable_source_a_sign(memory, status, detection, target_xy):
            continue
        if _is_probable_held_or_floor_reflection(memory, status, detection, target_xy):
            continue
        if _xy_in_zones(target_xy, memory.hazard_zones):
            continue
        candidates.append(detection)
    return sorted(
        candidates,
        key=lambda detection: (
            float(getattr(detection, "letter_score", 0.0)) * 120.0
            - abs(float(getattr(detection, "full_bearing_deg", getattr(detection, "angle_deg", 0.0)))) / 3.0
            + min(float(getattr(detection, "blob_area", 0)), 15000.0) / 1800.0
        ),
        reverse=True,
    )


def _remember_landmark_rays(observation: Observation, memory: AgentMemory) -> None:
    status = observation.robot_status
    if _status_unavailable(status):
        return
    rx, ry, _ = _robot_xy_yaw(status)
    for detection in observation.detections:
        color = getattr(detection, "color", None)
        if color not in COLOR_ORDER or not _looks_like_landmark_only(detection):
            continue
        ray = {
            "origin_x": rx,
            "origin_y": ry,
            "bearing_deg": _bearing_world_deg(status, detection),
            "area": float(getattr(detection, "blob_area", 0)),
            "letter": float(getattr(detection, "letter_score", 0.0)),
            "wood": float(getattr(detection, "wood_score", 0.0)),
        }
        memory.landmark_rays.setdefault(color, []).append(ray)
        memory.landmark_rays[color] = memory.landmark_rays[color][-8:]
        _map_event(memory, "landmark_ray", color=color, ray=ray)


def _remember_pad_estimates(observation: Observation, memory: AgentMemory) -> None:
    _remember_landmark_rays(observation, memory)
    if memory.held_color is None and memory.stage not in {"need_pad", "ready_place"}:
        return
    status = observation.robot_status
    if _status_unavailable(status):
        return
    if memory.held_color in COLOR_ORDER:
        colors = (memory.held_color,)
    else:
        colors = COLOR_ORDER
    for color in colors:
        if color in memory.confirmed_pad_xy:
            memory.known_pad_xy[color] = memory.confirmed_pad_xy[color]
            continue
        candidates = _pad_candidates(observation, memory, color)
        if DEBUG_PAD_REJECTS and not candidates and color == memory.held_color:
            debug_candidates = [
                detection
                for detection in observation.detections
                if detection.color == color and _looks_like_pad_shape_candidate(detection)
            ]
            debug_candidates = sorted(
                debug_candidates,
                key=lambda detection: (
                    float(getattr(detection, "letter_score", 0.0)) + float(getattr(detection, "wood_score", 0.0)),
                    getattr(detection, "blob_area", 0),
                ),
                reverse=True,
            )[:4]
            for debug_detection in debug_candidates:
                try:
                    debug_xy = _project_detection_xy(status, debug_detection, target_kind="pad")
                    unusable = _is_unusable_pad_xy(memory, color, debug_xy)
                    xy_text = f" xy=({debug_xy[0]:.2f},{debug_xy[1]:.2f}) unusable={unusable}"
                except Exception:
                    xy_text = ""
                print(
                    "[PadReject] "
                    f"color={color} area={debug_detection.blob_area} centroid={debug_detection.centroid} "
                    f"bbox={debug_detection.bbox} bearing={getattr(debug_detection, 'full_bearing_deg', debug_detection.angle_deg):.1f} "
                    f"letter={getattr(debug_detection, 'letter_score', 0.0):.3f} "
                    f"wood={getattr(debug_detection, 'wood_score', 0.0):.3f}{xy_text}"
                )
        if not candidates:
            continue
        detection = candidates[0]
        if color in memory.known_pad_xy and memory.held_color != color:
            continue
        target_xy = _project_detection_xy(status, detection, target_kind="pad")
        if _is_probable_source_a_sign(memory, status, detection, target_xy):
            _add_no_place_zone(memory, target_xy, reason="probable_A_source_sign", color=color, radius=0.75)
            continue
        if _is_probable_held_or_floor_reflection(memory, status, detection, target_xy):
            _map_event(memory, "pad_candidate_blocked", color=color, xy=target_xy, reason="held_or_floor_reflection")
            continue
        if _is_unusable_pad_xy(memory, color, target_xy):
            continue
        print(
            "[Target] pad "
            f"color={color} area={detection.blob_area} centroid={detection.centroid} "
            f"bbox={detection.bbox} bearing={getattr(detection, 'full_bearing_deg', detection.angle_deg):.1f} "
            f"letter={getattr(detection, 'letter_score', 0.0):.3f} "
            f"wood={getattr(detection, 'wood_score', 0.0):.3f} "
            f"xy=({target_xy[0]:.2f},{target_xy[1]:.2f})"
        )
        memory.known_pad_xy[color] = target_xy
        memory.candidate_pad_xy[color] = target_xy
        _map_event(memory, "drop_candidate", color=color, xy=target_xy, bbox=detection.bbox)


def _select_next_cube(observation: Observation, memory: AgentMemory) -> Any | None:
    candidates = _cube_candidates(observation, memory)
    if not candidates:
        return None
    best = candidates[0]
    if memory.priority_colors:
        for color in memory.priority_colors:
            for detection in candidates:
                if detection.color == color and _score_cube_candidate(detection, memory) >= _score_cube_candidate(best, memory) - 6.0:
                    return detection
    return best


def _is_failure(action_result: dict[str, Any] | None) -> bool:
    return bool(action_result and action_result.get("ok") is False)


def _choose_local_decision(
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None,
) -> AgentDecision:
    last_action_result = last_result.get("action_result") if last_result else None
    if memory.fallen_detected:
        return AgentDecision(
            next_action="stop",
            reason="Local Level 1 policy: robot is fallen; stop instead of repeating impossible actions.",
        )
    if _status_unavailable(observation.robot_status):
        return AgentDecision(
            next_action="search_pad" if memory.held_color else "search_cube",
            target_color=memory.held_color,
            reason="Local Level 1 policy: robot_status unavailable; rescan without coordinate navigation.",
            recovery_strategy="state_retry",
        )
    if memory.held_color:
        memory.active_color = memory.held_color
        if memory.blocked_colors.get(memory.held_color, 0) > 0:
            return AgentDecision(
                next_action="recover",
                target_color=memory.held_color,
                reason="Local Level 1 policy: held color recently failed score verification; discard and resume source farming.",
                recovery_strategy="discard_held_at_source",
            )
        if (
            _is_failure(last_action_result)
            and last_action_result.get("action") in {"search_pad", "navigate_to_pad", "place_cube"}
            and memory.consecutive_rpc_failed_scans > 0
        ):
            return AgentDecision(
                next_action="recover",
                target_color=memory.held_color,
                reason="Local Level 1 policy: pad-side vision/control RPC failed; recover before retrying.",
                recovery_strategy="back_up_and_rescan",
            )
        if memory.stage == "ready_place":
            return AgentDecision(
                next_action="place_cube",
                target_color=memory.held_color,
                reason="Local Level 1 policy: pad coordinate navigation succeeded.",
            )
        if memory.held_color not in memory.known_pad_xy:
            return AgentDecision(
                next_action="search_pad",
                target_color=memory.held_color,
                reason="Local Level 1 policy: holding cube but target pad coordinate is not visible yet.",
                recovery_strategy="wide_scan_then_turn",
            )
        return AgentDecision(
            next_action="navigate_to_pad",
            target_color=memory.held_color,
            reason="Local Level 1 policy: holding cube; use remembered/scanned pad coordinate.",
        )

    if memory.stage == "ready_pick" and memory.active_color:
        return AgentDecision(
            next_action="pick_cube",
            target_color=memory.active_color,
            reason="Local Level 1 policy: cube coordinate navigation succeeded.",
        )
    if memory.stage == "ready_pick" and memory.active_target_kind == "source":
        return AgentDecision(
            next_action="pick_cube",
            reason="Local Level 1 policy: returned to source pickup anchor.",
        )

    if memory.known_source_xy is not None:
        robot_xy = _robot_xy_yaw(observation.robot_status)[:2]
        source_distance = _xy_distance(robot_xy, memory.known_source_xy)
        source_pick_cooling_down = memory.source_pick_cooldown > 0
        if source_pick_cooling_down:
            memory.source_pick_cooldown -= 1
        if source_distance > SOURCE_REVISIT_RADIUS_M:
            memory.active_color = None
            memory.active_target_xy = memory.known_source_xy
            memory.active_target_kind = "source"
            print(
                "[Target] source "
                f"xy=({memory.known_source_xy[0]:.2f},{memory.known_source_xy[1]:.2f}) "
                f"robot=({robot_xy[0]:.2f},{robot_xy[1]:.2f})"
            )
            return AgentDecision(
                next_action="navigate_to_cube",
                reason="Local Level 1 policy: return to remembered source pickup anchor.",
            )
        memory.active_color = None
        memory.active_target_xy = memory.known_source_xy
        memory.active_target_kind = "source"
        if source_pick_cooling_down:
            return AgentDecision(
                next_action="recover",
                reason="Local Level 1 policy: source pick just failed; nudge and rescan before retrying.",
                recovery_strategy="source_pick_nudge",
            )
        return AgentDecision(
            next_action="pick_cube",
            reason="Local Level 1 policy: at source anchor; pick nearest conveyor cube directly.",
        )

    if _source_pick_visible(observation):
        robot_xy = _robot_xy_yaw(observation.robot_status)[:2]
        memory.active_color = None
        memory.active_target_xy = robot_xy
        memory.active_target_kind = "source"
        return AgentDecision(
            next_action="pick_cube",
            reason="Local Level 1 policy: close conveyor cubes are visible; try source farming pick before slow coordinate navigation.",
        )

    target = _select_next_cube(observation, memory)
    if target is None:
        return AgentDecision(
            next_action="recover" if _is_failure(last_action_result) else "search_cube",
            reason="Local Level 1 policy: no plausible cube candidate visible.",
            recovery_strategy="scan_and_reposition",
        )
    memory.active_color = target.color
    memory.active_target_xy = _project_detection_xy(observation.robot_status, target, target_kind="cube")
    memory.active_target_kind = "cube"
    print(
        "[Target] cube "
        f"color={target.color} area={target.blob_area} centroid={target.centroid} "
        f"bbox={target.bbox} bearing={getattr(target, 'full_bearing_deg', target.angle_deg):.1f} "
        f"xy=({memory.active_target_xy[0]:.2f},{memory.active_target_xy[1]:.2f})"
    )
    return AgentDecision(
        next_action="navigate_to_cube",
        target_color=target.color,
        reason="Local Level 1 policy: projected visible source/cube blob to world coordinate.",
    )


async def _ask_llm_advice(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None,
    local_decision: AgentDecision,
) -> AgentDecision | None:
    disabled = os.environ.get("MENLO_LEVEL1_DISABLE_LLM", "").lower() in {"1", "true", "yes"}
    backend_url, api_key, model = _llm_backend_config()
    if backend_url and "tokamak.sh" in backend_url:
        import menlo_runner.llm
        menlo_runner.llm.OPENROUTER_URL = backend_url
    if disabled or not api_key:
        return None
    should_ask = LLM_ADVICE_EVERY_N_CYCLES > 0 and memory.cycle_index % LLM_ADVICE_EVERY_N_CYCLES == 0
    if not should_ask:
        return None

    from menlo_runner.llm import call_llm

    context = build_decision_context(task, observation, memory, last_result)
    context["local_recommendation"] = local_decision.__dict__
    context["known_pad_xy"] = memory.known_pad_xy
    messages = [
        {
            "role": "system",
            "content": (
                "You are the Level 1 high-level reviewer for a warehouse robot. "
                "Level 1 may use robot_status, camera observations, and go_to with coordinates estimated from observations. "
                "Do not use scene_state, exact entity IDs, or global map coordinates. "
                "Keep the local recommendation unless it clearly violates the task. "
                "Return only JSON: next_action, target_color, reason, optional recovery_strategy."
            ),
        },
        {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=str)},
    ]
    try:
        reply = await asyncio.wait_for(
            asyncio.to_thread(
                call_llm,
                messages,
                api_key=api_key,
                model=model,
                timeout_s=max(1, int(LLM_ADVICE_TIMEOUT_S)),
            ),
            timeout=LLM_ADVICE_TIMEOUT_S + 0.75,
        )
    except Exception as exc:
        memory.llm_notes.append(f"LLM unavailable: {type(exc).__name__}")
        return None
    advice = parse_agent_decision(reply or "")
    if advice is None:
        memory.llm_notes.append("LLM parse failed.")
        return None
    memory.llm_notes.append(advice.reason[:160])
    return advice


def _validate_decision(decision: AgentDecision, fallback: AgentDecision, memory: AgentMemory) -> AgentDecision:
    if decision.next_action not in ALLOWED_NEXT_ACTIONS:
        return fallback
    if decision.next_action != fallback.next_action:
        if not (decision.next_action == "recover" and fallback.next_action == "recover"):
            return fallback
    if decision.target_color is not None and decision.target_color not in COLOR_ORDER:
        return fallback
    return decision


# ---------------------------------------------------------------------------
# LEVEL 1 학생 TODO: coordinate-guided action 구현
# ---------------------------------------------------------------------------
# Level 1은 go_to를 사용할 수 있지만 observation으로 추정한 coordinate에만 사용할 수 있습니다.
# Entity ID, scene_state, ground-truth object coordinate를 사용하지 마세요.


def estimate_target_xy_from_observation(observation: Observation, target_color: str | None) -> tuple[float, float] | None:
    """Camera observation으로 target world coordinate를 추정합니다.

    Backward-compatible helper입니다. Fast implementation에서는 memory를 함께 쓰는
    execute_decision 경로에서 더 구체적으로 호출합니다.
    """
    if _status_unavailable(observation.robot_status):
        return None
    if target_color is None:
        candidates = [d for d in observation.detections if _looks_like_cube_candidate(d)]
        if not candidates:
            return None
        detection = max(candidates, key=lambda item: item.blob_area)
        return _project_detection_xy(observation.robot_status, detection, target_kind="cube")

    pad_candidates = [d for d in observation.detections if d.color == target_color and _looks_like_pad_candidate(d)]
    if pad_candidates:
        detection = max(pad_candidates, key=lambda item: item.blob_area)
        return _project_detection_xy(observation.robot_status, detection, target_kind="pad")

    cube_candidates = [d for d in observation.detections if d.color == target_color and _looks_like_cube_candidate(d)]
    if cube_candidates:
        detection = max(cube_candidates, key=lambda item: item.blob_area)
        return _project_detection_xy(observation.robot_status, detection, target_kind="cube")
    return None


async def go_to_xy(ctx: Any, x: float, y: float) -> Any:
    """Coordinate-based go_to입니다. 학생 시스템이 추정한 x/y에만 사용하세요."""
    return await ctx.invoke(
        "go_to",
        {
            "target": {
                "kind": "pose",
                "pose": {"frame_id": "world", "position": [x, y, 0]},
            }
        },
        timeout_s=GO_TO_TIMEOUT_S,
    )


async def _supervised_go_to_xy(
    ctx: Any,
    memory: AgentMemory,
    target_xy: tuple[float, float],
    *,
    action: str,
    tolerance_m: float,
) -> dict[str, Any]:
    """Run coordinate go_to, then verify actual robot position before declaring failure."""
    started = time.perf_counter()
    result_summary_data: dict[str, Any] | None = None
    error_text: str | None = None
    sdk_ok = False
    try:
        result = await go_to_xy(ctx, *target_xy)
        sdk_ok = _sdk_ok(result)
        result_summary_data = result_summary(result)
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"

    robot_xy: tuple[float, float] | None = None
    robot_z: float | None = None
    status_warning: str | None = None
    try:
        status = await get_robot_status(ctx)
        robot_xy = _robot_xy_yaw(status)[:2]
        robot_z = _robot_z(status)
        memory.last_robot_status = status
        memory.last_robot_xy = robot_xy
        memory.last_robot_z = robot_z
    except Exception as exc:
        status_warning = f"{type(exc).__name__}: {exc}"

    if not sdk_ok and robot_xy is not None:
        dist = _xy_distance(robot_xy, target_xy)
        if dist > tolerance_m:
            dx = target_xy[0] - robot_xy[0]
            dy = target_xy[1] - robot_xy[1]
            angle = math.atan2(dy, dx)
            backup_xy = (target_xy[0] - math.cos(angle) * 0.5, target_xy[1] - math.sin(angle) * 0.5)
            print(f"[Navigate] go_to failed. Attempting backup waypoint xy={backup_xy} to avoid obstacle.")
            try:
                result2 = await go_to_xy(ctx, *backup_xy)
                sdk_ok = _sdk_ok(result2)
                result_summary_data = result_summary(result2)
                status = await get_robot_status(ctx)
                robot_xy = _robot_xy_yaw(status)[:2]
                robot_z = _robot_z(status)
                memory.last_robot_status = status
                memory.last_robot_xy = robot_xy
                memory.last_robot_z = robot_z
            except Exception as exc:
                print(f"[Navigate] Backup go_to failed: {exc}")


    reached = robot_xy is not None and _xy_distance(robot_xy, target_xy) <= tolerance_m
    if robot_xy is not None and robot_z is not None and not _robot_height_safe(robot_z):
        _add_hazard_zone(memory, robot_xy, reason="unsafe_robot_height_after_go_to")
    result_error = ""
    if result_summary_data:
        result_error = str(result_summary_data.get("error") or "")
    failure_text = f"{result_error} {error_text or ''}".lower()
    pad_stuck = action == "navigate_to_pad" and "stuck" in failure_text
    corrected_by_position = reached and not sdk_ok and not pad_stuck
    if error_text and not reached:
        try:
            await asyncio.wait_for(cancel_action(ctx), timeout=3.0)
        except Exception:
            pass

    return {
        "action": action,
        "ok": sdk_ok or corrected_by_position,
        "target_xy": target_xy,
        "robot_xy": robot_xy,
        "robot_z": robot_z,
        "reached_by_position": reached,
        "corrected_by_position": corrected_by_position,
        "blocked_by_stuck": pad_stuck,
        "tolerance_m": tolerance_m,
        "duration_s": round(time.perf_counter() - started, 3),
        "result": result_summary_data,
        "error": error_text,
        "status_warning": status_warning,
    }


async def execute_decision(
    ctx: Any,
    decision: AgentDecision,
    observation: Observation,
    memory: AgentMemory,
) -> dict[str, Any]:
    """검증된 LLM 결정 하나를 Level 1 robot 행동으로 변환합니다.

    Coordinate go_to는 오직 camera observation/robot_status에서 추정한 좌표에만 씁니다.
    """
    if decision.next_action == "search_cube":
        detections = await scan_head(ctx, yaws=(-0.9, -0.45, 0.0, 0.45, 0.9), pitch=CUBE_NAV_PITCH, memory=memory)
        return {"action": "search_cube", "ok": True, "status": "scanned", "detections": len(detections)}

    if decision.next_action == "search_pad":
        target_color = decision.target_color or memory.held_color
        detections = await scan_head(ctx, yaws=PAD_SCAN_YAWS, pitch=PAD_NAV_PITCH, memory=memory)
        if memory.last_scan_attempts and memory.last_scan_failures >= memory.last_scan_attempts:
            return {
                "action": "search_pad",
                "ok": False,
                "status": "vision_rpc_failed",
                "target_color": target_color,
                "detections": len(detections),
                "scan_failures": memory.last_scan_failures,
            }
        status = await get_robot_status_safe(ctx, memory)
        if _status_unavailable(status):
            return {
                "action": "search_pad",
                "ok": False,
                "status": "robot_status_unavailable",
                "target_color": target_color,
                "detections": len(detections),
            }
        temp_observation = Observation(robot_status=status, detections=detections)
        _remember_pad_estimates(temp_observation, memory)
        if target_color:
            candidates = _pad_candidates(temp_observation, memory, target_color)
            for candidate in candidates:
                target_xy = _project_detection_xy(status, candidate, target_kind="pad")
                if _is_probable_source_a_sign(memory, status, candidate, target_xy):
                    _add_no_place_zone(memory, target_xy, reason="probable_A_source_sign", color=target_color, radius=0.75)
                    continue
                if _is_probable_held_or_floor_reflection(memory, status, candidate, target_xy):
                    _map_event(memory, "pad_candidate_blocked", color=target_color, xy=target_xy, reason="held_or_floor_reflection")
                    continue
                block_reason = _navigation_block_reason(memory, target_xy, purpose="pad", color=target_color)
                if _is_unusable_pad_xy(memory, target_color, target_xy) or block_reason:
                    if block_reason:
                        _map_event(memory, "pad_candidate_blocked", color=target_color, xy=target_xy, reason=block_reason)
                    continue
                print(
                    "[Target] pad "
                    f"color={target_color} area={candidate.blob_area} centroid={candidate.centroid} "
                    f"bbox={candidate.bbox} bearing={getattr(candidate, 'full_bearing_deg', candidate.angle_deg):.1f} "
                    f"letter={getattr(candidate, 'letter_score', 0.0):.3f} "
                    f"wood={getattr(candidate, 'wood_score', 0.0):.3f} "
                    f"xy=({target_xy[0]:.2f},{target_xy[1]:.2f})"
                )
                memory.known_pad_xy[target_color] = target_xy
                memory.candidate_pad_xy[target_color] = target_xy
                memory.target_pad_xy = target_xy
                _map_event(memory, "drop_candidate", color=target_color, xy=target_xy, bbox=candidate.bbox)
                return {
                    "action": "search_pad",
                    "ok": True,
                    "status": "found",
                    "target_xy": target_xy,
                    "target_color": target_color,
                    "detections": len(detections),
                }
            landmark_candidates = _landmark_seek_candidates(temp_observation, memory, target_color)
            for landmark in landmark_candidates:
                seek_xy = _project_detection_xy(status, landmark, target_kind="pad")
                robot_xy = _robot_xy_yaw(status)[:2]
                nav_xy, partial = _bounded_step_xy(robot_xy, seek_xy, max_step_m=UNCONFIRMED_PAD_MAX_STEP_M)
                block_reason = _navigation_block_reason(memory, nav_xy, purpose="scan", color=target_color)
                if block_reason:
                    _map_event(memory, "landmark_seek_blocked", color=target_color, xy=nav_xy, reason=block_reason)
                    continue
                _map_event(memory, "landmark_seek", color=target_color, xy=seek_xy, nav_xy=nav_xy, bbox=landmark.bbox)
                nav_result = await _supervised_go_to_xy(
                    ctx,
                    memory,
                    nav_xy,
                    action="landmark_seek",
                    tolerance_m=NAV_REACHED_TOLERANCE_M,
                )
                if nav_result.get("ok") is True:
                    memory.known_pad_xy[target_color] = seek_xy
                    memory.candidate_pad_xy[target_color] = seek_xy
                    memory.target_pad_xy = seek_xy
                return {
                    "action": "search_pad",
                    "ok": nav_result.get("ok") is True,
                    "status": "landmark_seek_go_to",
                    "target_color": target_color,
                    "target_xy": seek_xy,
                    "nav_xy": nav_xy,
                    "partial": partial,
                    "detections": len(detections),
                    "nav_result": nav_result,
                }
        motion, search_status = _pad_search_motion(memory)
        try:
            await move_velocity(ctx, **motion)
        except Exception as exc:
            return {
                "action": "search_pad",
                "ok": False,
                "status": "turn_failed",
                "target_color": target_color,
                "error": f"{type(exc).__name__}: {exc}",
                "detections": len(detections),
            }
        return {
            "action": "search_pad",
            "ok": True,
            "status": search_status,
            "target_color": target_color,
            "detections": len(detections),
            "motion": motion,
        }

    if decision.next_action == "navigate_to_cube":
        if _status_unavailable(observation.robot_status):
            return {"action": "navigate_to_cube", "ok": False, "reason": "robot_status unavailable"}
        target_xy = memory.active_target_xy
        if target_xy is None:
            target_xy = estimate_target_xy_from_observation(observation, decision.target_color)
        if target_xy is None:
            return {"action": "navigate_to_cube", "ok": False, "reason": "coordinate estimate 없음"}
        nav_result = await _supervised_go_to_xy(
            ctx,
            memory,
            target_xy,
            action="navigate_to_cube",
            tolerance_m=NAV_REACHED_TOLERANCE_M,
        )
        nav_result["target_color"] = decision.target_color
        return nav_result

    if decision.next_action == "navigate_to_pad":
        if _status_unavailable(observation.robot_status):
            return {"action": "navigate_to_pad", "ok": False, "reason": "robot_status unavailable"}
        target_xy = memory.known_pad_xy.get(decision.target_color or "")
        if target_xy is None:
            candidates = _pad_candidates(observation, memory, decision.target_color or "")
            if candidates:
                target_xy = _project_detection_xy(observation.robot_status, candidates[0], target_kind="pad")
                color_key = decision.target_color or ""
                if _is_probable_source_a_sign(memory, observation.robot_status, candidates[0], target_xy):
                    block_reason = "probable_A_source_sign"
                    _add_no_place_zone(memory, target_xy, reason=block_reason, color=color_key, radius=0.75)
                elif _is_probable_held_or_floor_reflection(memory, observation.robot_status, candidates[0], target_xy):
                    block_reason = "held_or_floor_reflection"
                else:
                    block_reason = _navigation_block_reason(memory, target_xy, purpose="pad", color=color_key)
                if block_reason:
                    _map_event(memory, "pad_candidate_blocked", color=color_key, xy=target_xy, reason=block_reason)
                    target_xy = None
                else:
                    memory.known_pad_xy[color_key] = target_xy
                    memory.candidate_pad_xy[color_key] = target_xy
        if target_xy is None:
            return {"action": "navigate_to_pad", "ok": False, "reason": "pad coordinate estimate 없음"}
        block_reason = _navigation_block_reason(memory, target_xy, purpose="pad", color=decision.target_color)
        if block_reason:
            if decision.target_color:
                memory.known_pad_xy.pop(decision.target_color, None)
                memory.candidate_pad_xy.pop(decision.target_color, None)
            return {"action": "navigate_to_pad", "ok": False, "target_xy": target_xy, "reason": block_reason}
        memory.target_pad_xy = target_xy
        robot_xy = _robot_xy_yaw(observation.robot_status)[:2]
        
        # Calculate standoff coordinate 0.65m away to avoid colliding with the pallet
        import math
        dx = robot_xy[0] - target_xy[0]
        dy = robot_xy[1] - target_xy[1]
        dist = math.hypot(dx, dy)
        if dist > 0.65:
            standoff_xy = (target_xy[0] + (dx/dist)*0.65, target_xy[1] + (dy/dist)*0.65)
        else:
            standoff_xy = robot_xy

        confirmed = decision.target_color in memory.confirmed_pad_xy
        max_step = MAX_PAD_NAV_STEP_M if confirmed else min(MAX_PAD_NAV_STEP_M, UNCONFIRMED_PAD_MAX_STEP_M)
        nav_xy, partial = _bounded_step_xy(robot_xy, standoff_xy, max_step_m=max_step)
        nav_block = _navigation_block_reason(memory, nav_xy, purpose="pad", color=decision.target_color)
        if nav_block:
            return {"action": "navigate_to_pad", "ok": False, "target_xy": target_xy, "nav_xy": nav_xy, "reason": nav_block}
        nav_result = await _supervised_go_to_xy(
            ctx,
            memory,
            nav_xy,
            action="navigate_to_pad",
            tolerance_m=NAV_REACHED_TOLERANCE_M if partial else PAD_REACHED_TOLERANCE_M,
        )
        nav_result["target_xy"] = target_xy
        nav_result["nav_xy"] = nav_xy
        nav_result["partial"] = partial
        nav_result["target_color"] = decision.target_color
        return nav_result

    if decision.next_action == "pick_cube":
        try:
            await set_head(ctx, yaw=0.0, pitch=CLOSE_LOOK_PITCH)
        except Exception:
            pass
        try:
            result = await pick_nearest_cube(ctx)
        except Exception as exc:
            return {"action": "pick_cube", "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {
            "action": "pick_cube",
            "target_color": decision.target_color,
            "ok": _sdk_ok(result),
            "result": result_summary(result),
        }

    if decision.next_action == "place_cube":
        was_holding = memory.held_color is not None
        target_color = decision.target_color or memory.held_color
        target_xy = memory.target_pad_xy or memory.known_pad_xy.get(target_color or "")
        close_observation = await close_visual_observation(ctx, memory, yaw=0.0, pitch=CLOSE_LOOK_PITCH)
        certificate = _make_pallet_certificate(close_observation, memory, target_color, target_xy=target_xy)
        if certificate is None:
            return {
                "action": "place_cube",
                "target_color": target_color,
                "was_holding": was_holding,
                "ok": False,
                "attempted_place": False,
                "target_xy": target_xy,
                "error": "missing NearPalletCertificate",
                "close_detections": len(close_observation.detections),
            }
        try:
            result = await place_nearest_zone(ctx)
        except Exception as exc:
            return {"action": "place_cube", "ok": False, "was_holding": was_holding, "error": f"{type(exc).__name__}: {exc}"}
        return {
            "action": "place_cube",
            "target_color": target_color,
            "was_holding": was_holding,
            "ok": _sdk_ok(result),
            "result": result_summary(result),
            "certificate": certificate,
        }

    if decision.next_action == "recover":
        if decision.recovery_strategy == "discard_held_at_source":
            if memory.known_source_xy is not None and not _status_unavailable(observation.robot_status):
                robot_xy = _robot_xy_yaw(observation.robot_status)[:2]
                if _xy_distance(robot_xy, memory.known_source_xy) > 0.75:
                    nav_result = await _supervised_go_to_xy(
                        ctx,
                        memory,
                        memory.known_source_xy,
                        action="recover",
                        tolerance_m=NAV_REACHED_TOLERANCE_M,
                    )
                    nav_result["status"] = "returning_to_source_to_discard"
                    nav_result["recovery_strategy"] = decision.recovery_strategy
                    return nav_result
            was_holding = memory.held_color is not None
            try:
                await set_head(ctx, yaw=0.0, pitch=CLOSE_LOOK_PITCH)
            except Exception:
                pass
            try:
                result = await place_nearest_zone(ctx)
            except Exception as exc:
                return {
                    "action": "recover",
                    "ok": False,
                    "status": "discard_failed",
                    "was_holding": was_holding,
                    "recovery_strategy": decision.recovery_strategy,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            return {
                "action": "recover",
                "ok": _sdk_ok(result),
                "status": "discard_attempt",
                "was_holding": was_holding,
                "recovery_strategy": decision.recovery_strategy,
                "result": result_summary(result),
            }
        if decision.recovery_strategy == "source_pick_nudge":
            side = 0.10 if memory.search_turns % 2 == 0 else -0.10
            turn = -0.22 if memory.search_turns % 2 == 0 else 0.22
            memory.search_turns += 1
            try:
                await set_head(ctx, yaw=0.0, pitch=CLOSE_LOOK_PITCH)
            except Exception:
                pass
            try:
                result = await move_velocity(ctx, vx=-0.10, vy=side, wz=turn, duration_s=SOURCE_NUDGE_DURATION_S)
            except Exception as exc:
                return {"action": "recover", "ok": False, "status": "source_nudge_failed", "error": f"{type(exc).__name__}: {exc}"}
            return {
                "action": "recover",
                "ok": _sdk_ok(result),
                "status": "source_nudged",
                "recovery_strategy": decision.recovery_strategy,
                "result": result_summary(result),
            }
        if memory.held_color and memory.held_color in memory.known_pad_xy:
            target_xy = memory.known_pad_xy[memory.held_color]
            nav_result = await _supervised_go_to_xy(
                ctx,
                memory,
                target_xy,
                action="recover",
                tolerance_m=PAD_REACHED_TOLERANCE_M,
            )
            return nav_result
        try:
            result = await move_velocity(ctx, vx=-0.16, wz=0.25, duration_s=0.9)
        except Exception as exc:
            return {"action": "recover", "ok": False, "status": "recover_velocity_failed", "error": f"{type(exc).__name__}: {exc}"}
        return {"action": "recover", "ok": _sdk_ok(result), "status": "stepped_back_and_turned", "result": result_summary(result)}

    if decision.next_action == "skip_target":
        if memory.active_color and memory.active_color not in memory.skipped_colors:
            memory.skipped_colors.append(memory.active_color)
        return {"action": "skip_target", "ok": True, "status": "skipped"}

    return {"action": decision.next_action, "ok": False, "status": "no_op"}


async def run_agent(
    ctx: Any,
    *,
    max_cycles: int = 20,
    completion: CompletionConfig | None = None,
) -> AgentMemory:
    """얇은 observe-LLM-act loop입니다. 이 loop만이 아니라 TODO 함수들을 수정하세요."""
    memory = AgentMemory()
    last_result: dict[str, Any] | None = None
    tracker = CompletionTracker(completion) if completion is not None else None
    await wait_for_runtime_ready(ctx, memory)

    for cycle in range(1, max_cycles + 1):
        memory.cycle_index = cycle
        print(f"\n[Level 1] Cycle {cycle}")
        if tracker is not None:
            first_cycle = tracker.started_at is None
            tracker.start_first_cycle()
            if first_cycle:
                tracker.print_start()
            try:
                reason = await asyncio.wait_for(tracker.stop_reason_from_scene(ctx), timeout=ROBOT_STATUS_TIMEOUT_S)
            except Exception as exc:
                print(f"[Progress Warning] completion check skipped: {type(exc).__name__}: {exc}")
                reason = None
            if reason is not None:
                tracker.mark_ended(reason)
                print(f"Completion target reached before cycle action: {reason}.")
                break

        observation = await observe_world(ctx, memory)
        decision = await decide_next_action(TASK, observation, memory, last_result)
        print("Agent decision:", decision)

        if decision.next_action == "stop":
            break

        started = time.perf_counter()
        action_result = await execute_decision(ctx, decision, observation, memory)
        action_duration = time.perf_counter() - started
        action_result.setdefault("duration_s", round(action_duration, 3))
        print(f"Action result ({action_duration:.2f}s): {action_result}")
        verified = await verify_outcome(ctx, decision, action_result)
        update_memory(memory, observation, decision, verified)
        print(
            f"[Progress] delivered_raw={memory.delivered_count} "
            f"scored_delta={_scored_delta(memory)} "
            f"held={memory.held_color} stage={memory.stage} "
            f"robot_xy={memory.last_robot_xy}"
        )
        last_result = verified
        if tracker is not None:
            try:
                reason = await asyncio.wait_for(tracker.stop_reason_from_scene(ctx), timeout=ROBOT_STATUS_TIMEOUT_S)
            except Exception as exc:
                print(f"[Progress Warning] completion check skipped: {type(exc).__name__}: {exc}")
                reason = None
            if reason is not None:
                tracker.mark_ended(reason)
                print(f"Completion target reached after cycle action: {reason}.")
                break

    if tracker is not None:
        try:
            await asyncio.wait_for(tracker.print_summary_from_scene(ctx), timeout=ROBOT_STATUS_TIMEOUT_S)
        except Exception as exc:
            print(f"[Progress Warning] final completion summary unavailable: {type(exc).__name__}: {exc}")
    return memory


async def run(ctx: Any) -> None:
    print(TASK)
    print("Level 1 fast coordinate-navigation agent 실행")
    viewer_url = getattr(ctx, "viewer_url", None)
    if viewer_url:
        try:
            Path("run_logs").mkdir(parents=True, exist_ok=True)
            Path("run_logs/latest_level1_url.txt").write_text(str(viewer_url), encoding="ascii")
            print("[Viewer] wrote run_logs/latest_level1_url.txt")
        except Exception as exc:
            print(f"[Viewer Warning] could not write viewer URL: {type(exc).__name__}: {exc}")
    memory = await run_agent(
        ctx,
        max_cycles=10_000,
        completion=CompletionConfig(level=1, max_elapsed_s=600),
    )
    print("\n실행 완료.")
    print(f"Delivered count: {memory.delivered_count} (scored_delta={_scored_delta(memory)})")
    print("Logs:")
    for item in memory.logs:
        print(item)



