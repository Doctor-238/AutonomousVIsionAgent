from __future__ import annotations

"""Menlo AI 로봇 분류 챌린지용 Level 0 프로젝트 시작 파일입니다.

이 파일은 완성된 해답이 아니라 시작 파일입니다.

지원 코드 섹션은 반복해서 작성할 필요가 없는 작은 래퍼와 자료 구조를 제공합니다.
학생 TODO 섹션은 팀의 프로젝트 설계를 직접 구현하는 부분입니다.

Level 0 규칙: scene_state, 정확한 entity ID, entity-target go_to를 사용할 수 있습니다.
핵심 과제는 고정 script가 아니라 의미 있는 LLM 보조 상위 단계 결정 loop를 구현하는 것입니다.
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

from menlo_runner.completion import CompletionConfig, CompletionTracker
from menlo_runner.scene import COLOR_TO_PAD


# ---------------------------------------------------------------------------
# 지원 코드: 공통 과제 정의와 필수 LLM 결정 형식
# ---------------------------------------------------------------------------
TASK = "Find and sort cubes from the source area into their matching destination pads."

COLOR_ORDER = ("red", "green", "blue", "yellow")
SOURCE_PAD_ID = "pad_A"
DEFAULT_PRIORITY_COLORS = ("red", "blue")
SOURCE_PICK_RADIUS_M = float(os.environ.get("MENLO_LEVEL0_SOURCE_RADIUS_M", "4.5"))
SOURCE_LOOKAHEAD_M = float(os.environ.get("MENLO_LEVEL0_SOURCE_LOOKAHEAD_M", "1.25"))
GO_TO_TIMEOUT_S = float(os.environ.get("MENLO_LEVEL0_GOTO_TIMEOUT_S", "90"))
PICK_TIMEOUT_S = float(os.environ.get("MENLO_LEVEL0_PICK_TIMEOUT_S", "30"))
PLACE_TIMEOUT_S = float(os.environ.get("MENLO_LEVEL0_PLACE_TIMEOUT_S", "45"))
LLM_ADVICE_TIMEOUT_S = float(os.environ.get("MENLO_LEVEL0_LLM_TIMEOUT_S", "4.5"))
LLM_ADVICE_EVERY_N_CYCLES = int(os.environ.get("MENLO_LEVEL0_LLM_EVERY_N", "8"))
ENABLE_UNWANTED_CLEAR = os.environ.get("MENLO_LEVEL0_CLEAR_UNWANTED", "1").lower() not in {
    "0",
    "false",
    "no",
}
SIM_SPEED = float(os.environ.get("MENLO_SIM_SPEED", "1.0"))

DESTINATION_SIGN_RULES = {
    "red": "B",
    "green": "C",
    "blue": "D",
    "yellow": "E",
}

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
    target_entity_id: str | None = None
    reason: str = ""
    recovery_strategy: str | None = None


@dataclass
class AgentMemory:
    """observe-decide-act cycle 사이에 agent가 유지하는 상태입니다."""

    delivered_count: int = 0
    discarded_count: int = 0
    picked_count: int = 0
    held_color: str | None = None
    held_entity_id: str | None = None
    active_cube_id: str | None = None
    active_color: str | None = None
    active_mode: str = "deliver"
    target_pad_id: str | None = None
    stage: str = "need_cube"
    cube_ready: bool = False
    pad_ready: bool = False
    priority_colors: list[str] = field(default_factory=list)
    source_position: tuple[float, float] | None = None
    route_scores: dict[str, float] = field(default_factory=dict)
    failed_attempts: dict[str, int] = field(default_factory=dict)
    completed_cube_ids: list[str] = field(default_factory=list)
    skipped_cube_ids: list[str] = field(default_factory=list)
    discarded_cube_ids: list[str] = field(default_factory=list)
    recent_outcomes: list[dict[str, Any]] = field(default_factory=list)
    llm_notes: list[str] = field(default_factory=list)
    cycle_index: int = 0
    logs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Observation:
    """LLM과 실행 코드에 전달할 간결한 full-state 관찰입니다."""

    robot_status: Any
    visible_cubes: list[dict[str, Any]]
    held_cube: dict[str, str] | None
    delivered_cube_ids: list[str]
    color_to_pad: dict[str, str]
    note: str = ""


def parse_agent_decision(text: str) -> AgentDecision | None:
    """필수 구조화 LLM JSON 출력을 parse하고 validate합니다."""
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

    target_entity_id = data.get("target_entity_id")
    if target_entity_id is not None and not isinstance(target_entity_id, str):
        return None

    return AgentDecision(
        next_action=next_action,
        target_color=target_color,
        target_entity_id=target_entity_id,
        reason=str(data.get("reason", "")),
        recovery_strategy=data.get("recovery_strategy"),
    )


def build_decision_context(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full-state 정보를 LLM에 전달하기 좋은 간결한 text context로 변환합니다."""
    return {
        "task": task,
        "visible_cubes": observation.visible_cubes,
        "held_cube": observation.held_cube,
        "delivered_cube_ids": observation.delivered_cube_ids,
        "color_to_pad": observation.color_to_pad,
        "memory": {
            "delivered_count": memory.delivered_count,
            "discarded_count": memory.discarded_count,
            "held_color": memory.held_color,
            "held_entity_id": memory.held_entity_id,
            "active_cube_id": memory.active_cube_id,
            "active_color": memory.active_color,
            "active_mode": memory.active_mode,
            "target_pad_id": memory.target_pad_id,
            "stage": memory.stage,
            "priority_colors": memory.priority_colors,
            "route_scores": memory.route_scores,
            "failed_attempts": memory.failed_attempts,
            "completed_cube_ids": memory.completed_cube_ids,
            "skipped_cube_ids": memory.skipped_cube_ids,
            "recent_outcomes": memory.recent_outcomes[-6:],
        },
        "last_result": last_result,
        "note": observation.note,
    }


# ---------------------------------------------------------------------------
# 지원 코드: Level 0 SDK wrapper
# ---------------------------------------------------------------------------

async def get_robot_status(ctx: Any) -> Any:
    """Robot pose, motion status, neck state를 읽습니다."""
    return await ctx.state("robot_status")


async def observe_full_state(ctx: Any) -> Observation:
    """scene_state helper로 프로젝트 Level 0 관찰을 수집합니다."""
    scene = await ctx.state("scene_state")
    robot_status = await get_robot_status(ctx)
    robot_entity = scene.entities.get("robot")
    robot_position = (
        tuple(robot_entity.pose.position)
        if robot_entity is not None and getattr(robot_entity, "pose", None)
        else tuple(robot_status.robot.pose.position)
    )

    cubes: list[dict[str, Any]] = []
    held_dict: dict[str, str] | None = None
    delivered: list[str] = []
    for entity_id, entity in scene.entities.items():
        if not entity_id.startswith("cube_"):
            continue
        color = entity.state.get("color", "?") if entity.state else "?"
        if getattr(entity, "attached_to", None):
            held_dict = {"entity_id": entity_id, "color": color}
        if not getattr(entity, "visible", False):
            delivered.append(entity_id)
            continue
        pose = getattr(entity, "pose", None)
        if not pose:
            continue
        p = tuple(pose.position)
        cubes.append(
            {
                "entity_id": entity_id,
                "color": color,
                "position": p,
                "distance_from_robot": round(
                    math.hypot(p[0] - robot_position[0], p[1] - robot_position[1]), 2
                ),
            }
        )
    cubes.sort(key=lambda cube: cube["distance_from_robot"])
    return Observation(
        robot_status=robot_status,
        visible_cubes=cubes,
        held_cube=held_dict,
        delivered_cube_ids=delivered,
        color_to_pad=dict(COLOR_TO_PAD),
    )


async def go_to_entity(ctx: Any, entity_id: str) -> Any:
    """Level 0 entity-target navigation입니다."""
    return await ctx.invoke(
        "go_to",
        {"target": {"kind": "entity", "entity_id": entity_id}},
        timeout_s=GO_TO_TIMEOUT_S,
    )


async def pick_cube_by_id(ctx: Any, cube_id: str) -> Any:
    """충분히 가까이 navigation한 뒤 특정 cube entity를 pick합니다."""
    return await ctx.invoke(
        "pick_entity",
        {"target": {"kind": "entity", "entity_id": cube_id}},
        timeout_s=PICK_TIMEOUT_S,
    )


async def place_on_pad_by_id(ctx: Any, pad_id: str) -> Any:
    """들고 있는 cube를 특정 pad entity에 place합니다."""
    return await ctx.invoke(
        "place_entity",
        {"target": {"kind": "entity", "entity_id": pad_id}},
        timeout_s=PLACE_TIMEOUT_S,
    )


def result_summary(result: Any) -> dict[str, Any]:
    """SDK result를 log하기 쉬운 작은 dictionary로 변환합니다."""
    error = getattr(result, "error", None)
    status = getattr(result, "status", None)
    return {
        "status": str(status) if status is not None else None,
        "error": getattr(error, "message", None) if error else None,
    }


def _sdk_ok(result: Any) -> bool:
    status = getattr(result, "status", None)
    return str(status).lower() == "done"


def _xy(position: Any) -> tuple[float, float]:
    return float(position[0]), float(position[1])


def _entity_position(scene: Any, entity_id: str) -> tuple[float, float] | None:
    entity = scene.entities.get(entity_id)
    pose = getattr(entity, "pose", None) if entity is not None else None
    if pose is None:
        return None
    return _xy(pose.position)


def _distance_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _source_position(scene: Any) -> tuple[float, float] | None:
    pad_a = _entity_position(scene, SOURCE_PAD_ID)
    if pad_a is not None:
        return pad_a

    pool_positions: list[tuple[float, float]] = []
    for entity_id, entity in scene.entities.items():
        if not entity_id.startswith("cube_pool_") or not getattr(entity, "visible", False):
            continue
        pose = getattr(entity, "pose", None)
        if pose is not None:
            pool_positions.append(_xy(pose.position))
    if not pool_positions:
        return None
    return (
        sum(pos[0] for pos in pool_positions) / len(pool_positions),
        sum(pos[1] for pos in pool_positions) / len(pool_positions),
    )


def _segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return _distance_xy(point, start)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
    closest = (sx + t * dx, sy + t * dy)
    return _distance_xy(point, closest)


def _obstacle_penalty(scene: Any, start: tuple[float, float], end: tuple[float, float]) -> float:
    """Approximate route clutter without using a private navmesh."""
    penalty = 0.0
    for entity_id, entity in scene.entities.items():
        if (
            entity_id == "robot"
            or entity_id.startswith("cube_")
            or entity_id.startswith("pad_")
        ):
            continue
        pose = getattr(entity, "pose", None)
        if pose is None:
            continue
        point = _xy(pose.position)
        clearance = _segment_distance(point, start, end)
        if clearance < 0.85:
            penalty += 1.6 - clearance
    return min(penalty, 8.0)


def _route_cost(scene: Any, start: tuple[float, float], end: tuple[float, float]) -> float:
    dx = abs(start[0] - end[0])
    dy = abs(start[1] - end[1])
    return dx + dy + 0.15 * math.hypot(dx, dy) + _obstacle_penalty(scene, start, end)


def _parse_color_list(value: str | None) -> list[str]:
    if not value:
        return []
    colors: list[str] = []
    for raw in value.replace(";", ",").split(","):
        color = raw.strip().lower()
        if color in COLOR_ORDER and color not in colors:
            colors.append(color)
    return colors


def _choose_priority_colors(scene: Any) -> tuple[list[str], dict[str, float], tuple[float, float] | None]:
    override = _parse_color_list(os.environ.get("MENLO_LEVEL0_PRIORITY_COLORS"))
    source = _source_position(scene)
    if len(override) >= 2:
        return override[:2], {}, source

    ranked: list[tuple[float, str]] = []
    scores: dict[str, float] = {}
    if source is not None:
        for color, pad_id in COLOR_TO_PAD.items():
            pad_position = _entity_position(scene, pad_id)
            if pad_position is None:
                continue
            score = _route_cost(scene, source, pad_position)
            scores[color] = round(score, 3)
            ranked.append((score, color))

    if ranked:
        priority = [color for _, color in sorted(ranked)[:2]]
    else:
        priority = list(DEFAULT_PRIORITY_COLORS)
    if len(priority) < 2:
        for color in DEFAULT_PRIORITY_COLORS:
            if color not in priority:
                priority.append(color)
    return priority[:2], scores, source


def _cube_position(cube: dict[str, Any]) -> tuple[float, float] | None:
    position = cube.get("position")
    if position is None:
        return None
    return _xy(position)


def _cube_source_distance(cube: dict[str, Any], source: tuple[float, float] | None) -> float:
    pos = _cube_position(cube)
    if pos is None or source is None:
        return float(cube.get("distance_from_robot", 999.0))
    return _distance_xy(pos, source)


def _source_cubes(observation: Observation, memory: AgentMemory) -> list[dict[str, Any]]:
    pool = [
        cube
        for cube in observation.visible_cubes
        if str(cube.get("entity_id", "")).startswith("cube_pool_")
        and cube.get("entity_id") not in memory.completed_cube_ids
        and cube.get("entity_id") not in memory.discarded_cube_ids
    ]
    if not pool:
        pool = [
            cube
            for cube in observation.visible_cubes
            if cube.get("entity_id") not in memory.completed_cube_ids
            and cube.get("entity_id") not in memory.discarded_cube_ids
        ]

    if memory.source_position is not None:
        near_source = [
            cube
            for cube in pool
            if _cube_source_distance(cube, memory.source_position) <= SOURCE_PICK_RADIUS_M
        ]
        if near_source:
            pool = near_source

    return sorted(
        pool,
        key=lambda cube: (
            _cube_source_distance(cube, memory.source_position),
            float(cube.get("distance_from_robot", 999.0)),
        ),
    )


def _select_source_target(
    observation: Observation,
    memory: AgentMemory,
) -> tuple[dict[str, Any] | None, str]:
    source_cubes = _source_cubes(observation, memory)
    if not source_cubes:
        return None, "none"

    front = source_cubes[0]
    front_color = front.get("color")
    if front_color in memory.priority_colors:
        return front, "deliver"

    front_distance = _cube_source_distance(front, memory.source_position)
    for cube in source_cubes[1:]:
        if cube.get("color") not in memory.priority_colors:
            continue
        if _cube_source_distance(cube, memory.source_position) <= front_distance + SOURCE_LOOKAHEAD_M:
            return cube, "deliver"

    if ENABLE_UNWANTED_CLEAR:
        return front, "discard"
    return None, "wait"


def _held_target_pad_id(memory: AgentMemory) -> str | None:
    if memory.active_mode == "discard":
        return SOURCE_PAD_ID
    if memory.held_color is None:
        return None
    return COLOR_TO_PAD.get(memory.held_color)


def _result_failed(last_result: dict[str, Any] | None) -> bool:
    if not last_result:
        return False
    action_result = last_result.get("action_result", {})
    return action_result.get("ok") is False


def choose_fast_decision(
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> AgentDecision:
    """Fast high-level policy. LLM advice may validate it, but should not stall it."""
    held = observation.held_cube
    if held is not None:
        memory.held_entity_id = held["entity_id"]
        memory.held_color = held["color"]
        if memory.held_color not in memory.priority_colors:
            memory.active_mode = "discard"
        memory.target_pad_id = _held_target_pad_id(memory)
        if memory.stage == "ready_place":
            return AgentDecision(
                next_action="place_cube",
                target_color=memory.held_color,
                target_entity_id=memory.target_pad_id,
                reason="Fast policy: holding cube and target pad navigation is ready.",
            )
        return AgentDecision(
            next_action="navigate_to_pad",
            target_color=memory.held_color,
            target_entity_id=memory.target_pad_id,
            reason="Fast policy: holding cube; navigate to matching pad or source discard pad.",
        )

    memory.held_color = None
    memory.held_entity_id = None
    if memory.stage == "ready_pick" and memory.active_cube_id:
        return AgentDecision(
            next_action="pick_cube",
            target_color=memory.active_color,
            target_entity_id=memory.active_cube_id,
            reason="Fast policy: cube navigation already succeeded.",
        )

    target, mode = _select_source_target(observation, memory)
    if target is None:
        return AgentDecision(
            next_action="recover",
            reason=f"Fast policy: no usable source cube found (mode={mode}).",
            recovery_strategy="return_to_source",
        )

    memory.active_cube_id = str(target["entity_id"])
    memory.active_color = str(target["color"])
    memory.active_mode = mode
    memory.target_pad_id = COLOR_TO_PAD.get(memory.active_color) if mode == "deliver" else SOURCE_PAD_ID
    return AgentDecision(
        next_action="navigate_to_cube",
        target_color=memory.active_color,
        target_entity_id=memory.active_cube_id,
        reason=f"Fast policy: {mode} source cube from A/conveyor queue.",
    )


async def _ask_llm_advice(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None,
    fast_decision: AgentDecision,
) -> AgentDecision | None:
    disabled = os.environ.get("MENLO_LEVEL0_DISABLE_LLM", "").lower() in {"1", "true", "yes"}
    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("TOKAMAK_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )
    if disabled or not api_key:
        return None
    should_ask = (
        memory.cycle_index <= 1
        or _result_failed(last_result)
        or (
            LLM_ADVICE_EVERY_N_CYCLES > 0
            and memory.cycle_index % LLM_ADVICE_EVERY_N_CYCLES == 0
        )
    )
    if not should_ask:
        return None

    from menlo_runner.llm import call_llm

    context = build_decision_context(task, observation, memory, last_result)
    context["fast_recommendation"] = fast_decision.__dict__
    messages = [
        {
            "role": "system",
            "content": (
                "You are the high-level decision reviewer for a Level 0 warehouse robot. "
                "Level 0 may use scene_state, exact entity IDs, and entity go_to. "
                "The robot must pick cubes from the A/source conveyor side, not from random middle/opposite positions. "
                "The speed strategy is to deliver only the two priority colors with shortest route from source. "
                "Return ONLY JSON with next_action, target_color, target_entity_id, reason, and optional recovery_strategy. "
                "Allowed next_action values: search_cube, navigate_to_cube, pick_cube, search_pad, navigate_to_pad, place_cube, recover, skip_target, stop."
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
                model=os.environ.get("OPENROUTER_MODEL", "openrouter/free"),
                timeout_s=max(1, int(LLM_ADVICE_TIMEOUT_S)),
            ),
            timeout=LLM_ADVICE_TIMEOUT_S + 0.75,
        )
    except Exception as exc:
        memory.llm_notes.append(f"LLM advice unavailable: {type(exc).__name__}")
        return None

    advice = parse_agent_decision(reply or "")
    if advice is None:
        memory.llm_notes.append("LLM advice parse failed.")
        return None
    memory.llm_notes.append(advice.reason[:160])
    return advice


def _validate_or_fallback(
    decision: AgentDecision,
    fallback: AgentDecision,
    observation: Observation,
    memory: AgentMemory,
) -> AgentDecision:
    if decision.next_action not in ALLOWED_NEXT_ACTIONS:
        return fallback
    if decision.next_action != fallback.next_action:
        # Keep action cadence deterministic; LLM can confirm/adapt targets and explain recovery,
        # but should not replace a safe source-cycle action with a slow search loop.
        if not (decision.next_action == "recover" and fallback.next_action == "recover"):
            return fallback
    if decision.next_action in {"navigate_to_cube", "pick_cube"}:
        visible_ids = {str(cube["entity_id"]) for cube in observation.visible_cubes}
        if decision.target_entity_id not in visible_ids:
            return fallback
        if decision.target_color not in memory.priority_colors and memory.active_mode != "discard":
            return fallback
    if decision.next_action in {"navigate_to_pad", "place_cube"}:
        allowed_pad = _held_target_pad_id(memory)
        if allowed_pad is None:
            return fallback
        if decision.target_entity_id not in {None, allowed_pad}:
            return fallback
        decision.target_entity_id = allowed_pad
    if decision.next_action == "stop" and observation.visible_cubes:
        return fallback
    return decision


# ---------------------------------------------------------------------------
# 학생 TODO: LLM decision 함수
# ---------------------------------------------------------------------------

async def decide_next_action(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> AgentDecision:
    """다음 상위 단계 행동을 선택합니다.

    정상 cadence는 local fast policy가 맡고, LLM은 주기적/실패 시 advisory로만
    참여합니다. OpenRouter 무료 모델이 느려도 robot action cadence가 막히지 않게
    짧은 timeout 뒤 검증된 local decision으로 fallback합니다.
    """
    fast_decision = choose_fast_decision(observation, memory, last_result)
    advice = await _ask_llm_advice(task, observation, memory, last_result, fast_decision)
    if advice is None:
        return fast_decision
    return _validate_or_fallback(advice, fast_decision, observation, memory)


# ---------------------------------------------------------------------------
# 학생 TODO: observation, execution, verification, memory
# ---------------------------------------------------------------------------

async def observe_world(ctx: Any, memory: AgentMemory) -> Observation:
    """LLM과 실행 코드를 위해 현재 Level 0 관찰을 수집합니다.

    Level 0에서는 scene_state가 허용되므로, 시작 직후 A/source에서 각 pad까지의
    route cost를 추정해 가장 짧은 두 destination color만 반복 배송합니다.
    """
    observation = await observe_full_state(ctx)
    if not memory.priority_colors:
        scene = await ctx.state("scene_state")
        priority, scores, source = _choose_priority_colors(scene)
        memory.priority_colors = priority
        memory.route_scores = scores
        memory.source_position = source
        print(
            "[Init] Level 0 source strategy -> "
            f"priority_colors={memory.priority_colors}, route_scores={memory.route_scores}, "
            f"source={memory.source_position}"
        )
    observation.note = (
        f"priority_colors={memory.priority_colors}; source={memory.source_position}; "
        "only cube_pool/source-side cubes should be selected."
    )
    return observation


async def execute_decision(
    ctx: Any,
    decision: AgentDecision,
    observation: Observation,
    memory: AgentMemory,
) -> dict[str, Any]:
    """검증된 LLM 결정 하나를 Level 0 robot 행동으로 변환합니다.

    모든 SDK action은 target validation을 거친 high-level decision에서만 호출합니다.
    """
    if decision.next_action == "stop":
        return {"action": "stop", "ok": True, "status": "stopped"}

    if decision.next_action in {"search_cube", "search_pad"}:
        return {"action": decision.next_action, "ok": True, "status": "state_search_only"}

    if decision.next_action == "skip_target":
        if memory.active_cube_id and memory.active_cube_id not in memory.skipped_cube_ids:
            memory.skipped_cube_ids.append(memory.active_cube_id)
        memory.active_cube_id = None
        memory.active_color = None
        memory.cube_ready = False
        memory.stage = "need_cube"
        return {"action": "skip_target", "ok": True, "status": "skipped"}

    if decision.next_action == "recover":
        target = _held_target_pad_id(memory) if memory.held_color else SOURCE_PAD_ID
        try:
            result = await go_to_entity(ctx, target)
            summary = result_summary(result)
            return {
                "action": "recover",
                "target_entity_id": target,
                "ok": _sdk_ok(result),
                "result": summary,
                "recovery_strategy": decision.recovery_strategy,
            }
        except Exception as exc:
            return {
                "action": "recover",
                "target_entity_id": target,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    if decision.next_action == "navigate_to_cube":
        cube_id = decision.target_entity_id or memory.active_cube_id
        if not cube_id:
            return {"action": "navigate_to_cube", "ok": False, "error": "missing cube target"}
        result = await go_to_entity(ctx, cube_id)
        return {
            "action": "navigate_to_cube",
            "target_entity_id": cube_id,
            "target_color": decision.target_color,
            "mode": memory.active_mode,
            "ok": _sdk_ok(result),
            "result": result_summary(result),
        }

    if decision.next_action == "pick_cube":
        cube_id = decision.target_entity_id or memory.active_cube_id
        if not cube_id:
            return {"action": "pick_cube", "ok": False, "error": "missing cube target"}
        result = await pick_cube_by_id(ctx, cube_id)
        return {
            "action": "pick_cube",
            "target_entity_id": cube_id,
            "target_color": decision.target_color,
            "mode": memory.active_mode,
            "ok": _sdk_ok(result),
            "result": result_summary(result),
        }

    if decision.next_action == "navigate_to_pad":
        pad_id = decision.target_entity_id or _held_target_pad_id(memory)
        if not pad_id:
            return {"action": "navigate_to_pad", "ok": False, "error": "missing pad target"}
        result = await go_to_entity(ctx, pad_id)
        return {
            "action": "navigate_to_pad",
            "target_entity_id": pad_id,
            "target_color": decision.target_color,
            "mode": memory.active_mode,
            "ok": _sdk_ok(result),
            "result": result_summary(result),
        }

    if decision.next_action == "place_cube":
        pad_id = decision.target_entity_id or _held_target_pad_id(memory)
        if not pad_id:
            return {"action": "place_cube", "ok": False, "error": "missing pad target"}
        was_holding = observation.held_cube is not None
        result = await place_on_pad_by_id(ctx, pad_id)
        return {
            "action": "place_cube",
            "target_entity_id": pad_id,
            "target_color": decision.target_color,
            "mode": memory.active_mode,
            "was_holding": was_holding,
            "ok": _sdk_ok(result),
            "result": result_summary(result),
        }

    return {"action": decision.next_action, "ok": False, "error": "unsupported action"}


async def verify_outcome(ctx: Any, decision: AgentDecision, action_result: dict[str, Any]) -> dict[str, Any]:
    """마지막 action이 성공한 것처럼 보이는지 확인합니다.

    중요한 action 뒤에는 scene_state를 다시 읽어 held/delivered 상태를 확인합니다.
    """
    observation = await observe_full_state(ctx)
    return {
        "decision": decision.__dict__,
        "action_result": action_result,
        "held_cube": observation.held_cube,
        "delivered_cube_ids": observation.delivered_cube_ids,
        "delivered_scene_count": len(observation.delivered_cube_ids),
    }


def update_memory(
    memory: AgentMemory,
    observation: Observation,
    decision: AgentDecision,
    verified: dict[str, Any],
) -> None:
    """각 cycle 뒤 지속 상태를 update합니다.

    Active cube, held cube, delivery/discard count, failure, skip history를 추적합니다.
    """
    action_result = verified.get("action_result", {})
    action = action_result.get("action")
    ok = action_result.get("ok") is True
    held_after = verified.get("held_cube")
    memory.held_entity_id = held_after["entity_id"] if held_after else None
    memory.held_color = held_after["color"] if held_after else None

    if action == "navigate_to_cube":
        memory.cube_ready = ok
        memory.stage = "ready_pick" if ok else "need_cube"
        if not ok and memory.active_cube_id:
            memory.failed_attempts[memory.active_color or "unknown"] = (
                memory.failed_attempts.get(memory.active_color or "unknown", 0) + 1
            )
            if memory.active_cube_id not in memory.skipped_cube_ids:
                memory.skipped_cube_ids.append(memory.active_cube_id)
            memory.active_cube_id = None
            memory.active_color = None

    elif action == "pick_cube":
        if ok and held_after:
            memory.picked_count += 1
            memory.cube_ready = False
            memory.pad_ready = False
            actual_color = held_after["color"]
            memory.held_color = actual_color
            memory.held_entity_id = held_after["entity_id"]
            if actual_color not in memory.priority_colors:
                memory.active_mode = "discard"
            else:
                memory.active_mode = "deliver"
            memory.target_pad_id = _held_target_pad_id(memory)
            memory.stage = "need_pad"
        elif held_after:
            memory.stage = "need_pad"
        else:
            memory.cube_ready = False
            memory.stage = "need_cube"
            if memory.active_cube_id and memory.active_cube_id not in memory.skipped_cube_ids:
                memory.skipped_cube_ids.append(memory.active_cube_id)
            memory.active_cube_id = None
            memory.active_color = None

    elif action == "navigate_to_pad":
        memory.pad_ready = ok
        memory.stage = "ready_place" if ok else "need_pad"

    elif action == "place_cube":
        placed = ok and action_result.get("was_holding") and held_after is None
        placed_cube_id = memory.held_entity_id or observation.held_cube["entity_id"] if observation.held_cube else None
        if placed and action_result.get("mode") == "deliver":
            memory.delivered_count += 1
            if placed_cube_id and placed_cube_id not in memory.completed_cube_ids:
                memory.completed_cube_ids.append(placed_cube_id)
        elif placed and action_result.get("mode") == "discard":
            memory.discarded_count += 1
            if placed_cube_id and placed_cube_id not in memory.discarded_cube_ids:
                memory.discarded_cube_ids.append(placed_cube_id)

        if placed:
            memory.held_color = None
            memory.held_entity_id = None
            memory.active_cube_id = None
            memory.active_color = None
            memory.active_mode = "deliver"
            memory.target_pad_id = None
            memory.cube_ready = False
            memory.pad_ready = False
            memory.stage = "need_cube"
        else:
            memory.pad_ready = False
            memory.stage = "need_pad" if held_after else "need_cube"

    elif action == "recover":
        if memory.held_color:
            memory.stage = "need_pad"
        else:
            memory.stage = "need_cube"
            memory.cube_ready = False
            memory.pad_ready = False

    outcome = {
        "action": action,
        "target": decision.target_color,
        "target_entity_id": action_result.get("target_entity_id"),
        "mode": action_result.get("mode", memory.active_mode),
        "success": ok,
        "error": action_result.get("error") or action_result.get("result", {}).get("error"),
    }
    memory.recent_outcomes.append(outcome)
    memory.recent_outcomes = memory.recent_outcomes[-10:]

    memory.logs.append(
        {
            "observation": {
                "visible_cube_count": len(observation.visible_cubes),
                "held_cube": observation.held_cube,
                "delivered_count": memory.delivered_count,
                "discarded_count": memory.discarded_count,
            },
            "decision": decision.__dict__,
            "memory": {
                "stage": memory.stage,
                "priority_colors": memory.priority_colors,
                "active_cube_id": memory.active_cube_id,
                "active_color": memory.active_color,
                "active_mode": memory.active_mode,
                "held_color": memory.held_color,
                "target_pad_id": memory.target_pad_id,
                "route_scores": memory.route_scores,
                "recent_outcomes": memory.recent_outcomes[-4:],
                "llm_notes": memory.llm_notes[-3:],
            },
            "verified": verified,
        }
    )


async def try_set_sim_speed(ctx: Any) -> None:
    """Use the viewer speed skill only when the simulator exposes it."""
    if SIM_SPEED <= 1.0:
        return
    try:
        skills = await ctx.session.discover_skills()
        skill_names = {getattr(skill, "name", "") for skill in skills}
    except Exception as exc:
        print(f"[Speed] Skill discovery failed; keeping default sim speed: {exc}")
        return
    if "set_sim_speed" not in skill_names:
        return
    for args in ({"speed": SIM_SPEED}, {"sim_speed": SIM_SPEED}, {"value": SIM_SPEED}):
        try:
            result = await ctx.invoke("set_sim_speed", args, timeout_s=5)
            print(f"[Speed] set_sim_speed {args} -> {result_summary(result)}")
            if _sdk_ok(result):
                return
        except Exception as exc:
            print(f"[Speed] set_sim_speed {args} failed: {type(exc).__name__}: {exc}")


async def run_agent(
    ctx: Any,
    *,
    task: str = TASK,
    max_cycles: int = 24,
    completion: CompletionConfig | None = None,
) -> AgentMemory:
    """얇은 observe-LLM-act loop입니다. 이 loop만이 아니라 TODO 함수들을 수정하세요."""
    memory = AgentMemory()
    last_result: dict[str, Any] | None = None
    tracker = CompletionTracker(completion) if completion is not None else None

    for cycle in range(1, max_cycles + 1):
        memory.cycle_index = cycle
        print(f"\n[Level 0] Cycle {cycle}")
        if tracker is not None:
            first_cycle = tracker.started_at is None
            tracker.start_first_cycle()
            if first_cycle:
                tracker.print_start()
            reason = await tracker.stop_reason_from_scene(ctx)
            if reason is not None:
                tracker.mark_ended(reason)
                print(f"Completion target reached before cycle action: {reason}.")
                break

        observation = await observe_world(ctx, memory)
        decision = await decide_next_action(task, observation, memory, last_result)
        print("Agent decision:", decision)

        if decision.next_action == "stop":
            break

        started = time.perf_counter()
        action_result = await execute_decision(ctx, decision, observation, memory)
        print(f"Action result ({time.perf_counter() - started:.2f}s): {action_result}")
        verified = await verify_outcome(ctx, decision, action_result)
        update_memory(memory, observation, decision, verified)
        last_result = verified
        if tracker is not None:
            reason = await tracker.stop_reason_from_scene(ctx)
            if reason is not None:
                tracker.mark_ended(reason)
                print(f"Completion target reached after cycle action: {reason}.")
                break

    if tracker is not None:
        await tracker.print_summary_from_scene(ctx)
    return memory


async def run(ctx: Any) -> None:
    print(TASK)
    print("Level 0 fast source-conveyor agent 실행")
    await try_set_sim_speed(ctx)
    memory = await run_agent(
        ctx,
        max_cycles=10_000,
        completion=CompletionConfig(level=0, max_elapsed_s=600),
    )
    print("\n실행 완료.")
    print(f"Correct delivered count: {memory.delivered_count}")
    print(f"Source-cleared/discarded count: {memory.discarded_count}")
    print(f"Picked count: {memory.picked_count}")
    print(f"Priority colors: {memory.priority_colors}")
    print(f"Route scores: {memory.route_scores}")
    print("Logs:")
    for item in memory.logs:
        print(item)



