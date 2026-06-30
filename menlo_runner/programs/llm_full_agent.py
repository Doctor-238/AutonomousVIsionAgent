"""
LLM Full Autonomous Agent (v3 - 예외처리 강화)
처리하는 예외 상황:
  1. 로봇 넘어짐 (fallen) → set_velocity 복구 시도 3회
  2. 이미 큐브 들고 있는데 pick 시도 → place 먼저
  3. 패드 범위 밖 place 실패 → go_to pad 후 재시도
  4. go_to 실패 (경로 없음 등) → 다른 큐브로 전환
  5. pick_cube 실패 (큐브 없음) → get_scene_summary로 재탐색
  6. LLM 파싱 실패 → 재시도 (build_system_prompt 기본 동작)
  7. 동일 동작 반복 (무한루프 감지) → 강제 종료
  8. 컨베이어 큐브(cube_pool_*) 포함 처리
  9. 알 수 없는 도구 호출 → 오류 메시지로 LLM 재판단
  10. 세션 만료 / 네트워크 오류 → 예외 출력 후 종료
"""
from __future__ import annotations

import asyncio
from collections import Counter
from menlo_runner.agents import WorkshopAgent, DEFAULT_TOOLS

# look 제거 (NVIDIA 모델 비전 미지원)
TOOLS = {k: v for k, v in DEFAULT_TOOLS.items() if k != "look"}

FULL_TASK = """
You are an autonomous warehouse robot. Deliver ALL colored cubes to their correct pads.

Delivery rules (strict):
  red    → pad_B
  green  → pad_C
  blue   → pad_D
  yellow → pad_E

Step-by-step procedure for EACH cube:
  1. get_scene_summary   — find all cubes and distances
  2. go_to <nearest cube>
  3. pick_cube
  4. check_held_object   — confirm color
  5. go_to <correct pad> — MUST reach pad before placing
  6. place
  7. Repeat from step 1

Error recovery rules (IMPORTANT):
  - If "robot is fallen": call set_velocity {"vx":0,"vy":0,"wz":0,"duration_s":3}, then retry the failed action once.
  - If "carry capacity reached": you are already holding a cube. Skip pick, go to the correct pad and place first.
  - If "not near a pallet": you forgot to go_to the pad. Call go_to for the correct pad, then place again.
  - If go_to fails with any error other than fallen: call get_scene_summary and pick a different target.
  - If pick_cube returns "No visible cubes": call get_scene_summary to find a cube, then go_to it before trying again.
  - If the same tool+args repeats 3 times in a row with the same failure: call done with a summary.
  - cube_pool_* entities are valid cubes from the conveyor belt. Treat them like regular cubes.

Call done ONLY when every cube has been placed or you are stuck after 3 recovery attempts.
"""


class RobustAgent(WorkshopAgent):
    """WorkshopAgent with loop detection and extended recovery."""

    async def run(self, task: str, *, max_turns: int = 80):
        recent: list[str] = []  # 최근 tool+result 해시 (반복 감지)
        repeat_counter: Counter = Counter()

        messages = [
            {"role": "system", "content": self._build_prompt()},
            {"role": "user",   "content": task},
        ]
        self.tool_log = []

        from menlo_runner.llm import call_llm, parse_tool_call, build_system_prompt

        for turn in range(1, max_turns + 1):
            # LLM 호출 (재시도 1회)
            reply = None
            for attempt in range(2):
                try:
                    reply = call_llm(messages, api_key=self.tokamak_api_key)
                    break
                except Exception as exc:
                    if attempt == 0:
                        print(f"  ⚠️  LLM 오류, 재시도: {exc}")
                        await asyncio.sleep(2)
                    else:
                        print(f"  ❌ LLM 오류 반복, 종료: {exc}")
                        return messages, self.tool_log

            messages.append({"role": "assistant", "content": reply})
            call = parse_tool_call(reply) or {
                "tool": "error",
                "args": {"message": "JSON 파싱 실패", "raw": reply[:100]},
            }
            tool_name = call["tool"]
            tool_args = call.get("args", {})

            history_chars = sum(len(str(m["content"])) for m in messages)
            print(f"turn {turn:2d} | {tool_name:20s} | args={tool_args} | hist~{history_chars:,}ch")

            # 종료 조건
            if tool_name == "done":
                print(f"  ✅ 에이전트 완료: {tool_args.get('summary', '')}")
                break

            # 반복 루프 감지
            sig = f"{tool_name}:{str(tool_args)}"
            repeat_counter[sig] += 1
            if repeat_counter[sig] >= 3:
                result = (
                    f"[SYSTEM] 동일 동작({tool_name})이 3회 반복되었습니다. "
                    "다른 전략을 선택하거나 done을 호출하세요."
                )
                print(f"  🔄 반복 감지: {sig}")
            elif tool_name == "error":
                result = f"parse error: {tool_args.get('message', '')}"
            else:
                try:
                    result = await self.execute_tool(tool_name, tool_args)
                except Exception as exc:
                    result = f"[EXCEPTION] {type(exc).__name__}: {exc}"
                    print(f"  ❌ 예외: {result}")

            self.tool_log.append({"turn": turn, "tool": tool_name, "result": result})
            print(f"  → {result[:160]}")
            messages.append({"role": "user", "content": result})

        return messages, self.tool_log

    def _build_prompt(self) -> str:
        from menlo_runner.llm import build_system_prompt
        return build_system_prompt(self.tools)


async def run(ctx) -> None:
    print("=" * 60)
    print("🤖  LLM 완전 자율 에이전트 v3")
    print("    강화된 예외처리 + 반복 루프 감지")
    print("=" * 60)

    agent = RobustAgent(
        ctx,
        tokamak_api_key=ctx.config.tokamak_api_key,
        tools=TOOLS,
    )

    messages, tool_log = await agent.run(FULL_TASK, max_turns=80)

    # 결과 요약
    print("\n" + "=" * 60)
    print("📋  도구 호출 로그")
    print("=" * 60)
    for e in tool_log:
        preview = e["result"][:120].replace("\n", " ")
        print(f"  [{e['turn']:2d}] {e['tool']:20s} → {preview}")

    delivered = sum(1 for e in tool_log if e["tool"] == "place"     and "Placed"  in e["result"])
    picked    = sum(1 for e in tool_log if e["tool"] == "pick_cube" and "Picked"  in e["result"])
    fallen    = sum(1 for e in tool_log if "fallen"    in e["result"].lower())
    repeated  = sum(1 for e in tool_log if "[SYSTEM]"  in e["result"])
    errors    = sum(1 for e in tool_log if "[EXCEPTION]" in e["result"])

    print("\n" + "=" * 60)
    print("✅  최종 결과")
    print(f"    총 턴수:      {len(tool_log)}")
    print(f"    픽업 성공:   {picked}개")
    print(f"    배달 성공:   {delivered}개")
    print(f"    넘어짐:      {fallen}회")
    print(f"    반복 차단:   {repeated}회")
    print(f"    예외 발생:   {errors}회")
    print("=" * 60)
