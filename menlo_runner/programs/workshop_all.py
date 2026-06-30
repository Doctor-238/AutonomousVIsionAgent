"""
Workshop 2 + 3 + 4 통합 실행 스크립트
하나의 뷰어 세션에서 순서대로 진행합니다.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

# ── Workshop 2: Perception ───────────────────────────────────────────────────
from menlo_runner.perception import annotate_detections, detect_color_blobs, perceive

# ── Workshop 3: Navigation ───────────────────────────────────────────────────
from menlo_runner.navigation import my_go_to_global, my_go_to_visual

# ── Workshop 4: Agent ────────────────────────────────────────────────────────
from menlo_runner.agents import WorkshopAgent

SEPARATOR = "\n" + "=" * 60 + "\n"


async def run(ctx) -> None:

    # ────────────────────────────────────────────────────────────────
    # WORKSHOP 2: Perception (카메라 색상 감지)
    # ────────────────────────────────────────────────────────────────
    print(SEPARATOR + "📷  WORKSHOP 2: Perception" + SEPARATOR)

    jpeg = await ctx.get_vision("pov")
    detections = detect_color_blobs(jpeg)

    print("색상 블롭 감지 결과:")
    if not detections:
        print("  (감지 없음)")
    for item in detections:
        print(
            f"  {item.color:8s}: 각도={item.angle_deg:+.1f}°  "
            f"면적={item.blob_area}px²  중심={item.centroid}"
        )

    out = Path("outputs/workshop2-perception.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(annotate_detections(jpeg, detections))
    print(f"\n주석 이미지 저장: {out}")
    print("✅ Workshop 2 완료!\n")
    await asyncio.sleep(1)

    # ────────────────────────────────────────────────────────────────
    # WORKSHOP 3: Navigation (경로 이동)
    # ────────────────────────────────────────────────────────────────
    print(SEPARATOR + "🧭  WORKSHOP 3: Navigation" + SEPARATOR)

    print("[Part A] 글로벌 좌표 기반 → pad_C 이동")
    reached = await my_go_to_global(ctx, "pad_C", tolerance_m=0.8, max_iters=3)
    print(f"  결과: {'✅ 도달' if reached else '❌ 실패'}")

    print("\n[Part B] 비전 기반 → 가장 가까운 큐브 추적")
    obs = await perceive(ctx)
    if not obs:
        print("  감지된 큐브 없음. 계속 진행합니다.")
    else:
        target_color = next(iter(obs))
        print(f"  타겟 색상: {target_color}")
        reached = await my_go_to_visual(ctx, target_color)
        print(f"  결과: {'✅ 도달' if reached else '❌ 실패'}")

    print("✅ Workshop 3 완료!\n")
    await asyncio.sleep(1)

    # ────────────────────────────────────────────────────────────────
    # WORKSHOP 4: Agent (LLM 에이전트)
    # ────────────────────────────────────────────────────────────────
    print(SEPARATOR + "🤖  WORKSHOP 4: Agent (LLM)" + SEPARATOR)

    TASK = (
        "Use get_scene_summary to find a visible cube, go to it, pick it up, "
        "check what you are holding, and place it on the correct pad. "
        "Call done after one successful delivery or if you cannot continue."
    )

    print(f"과제: {TASK}\n")
    agent = WorkshopAgent(ctx, tokamak_api_key=ctx.config.tokamak_api_key)
    _messages, tool_log = await agent.run(TASK, max_turns=12)

    print("\n[도구 호출 로그]")
    for entry in tool_log:
        print(f"  turn {entry['turn']:2d} | {entry['tool']:20s} → {entry['result'][:80]}")

    print("\n✅ Workshop 4 완료!")
    print(SEPARATOR + "🎉  전체 워크샵 완료!" + SEPARATOR)
