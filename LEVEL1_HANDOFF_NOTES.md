# Level 1 Handoff Notes

이 문서는 다음 작업자가 Level 1 에이전트를 이어받을 때 반드시 지켜야 할 요구사항과 실전 노하우를 정리한 것이다. 프로젝트 전체 설명은 기존 문서를 알고 있다고 가정하고, 여기서는 현재 코드가 실패했던 지점과 현재 Level 1 제출 전략만 기록한다.

## 현재 목표

- 제출 레벨은 Level 1이다.
- 목표는 10분 동안 정확한 배달 수를 늘리는 것이다.
- 속도는 중요하지만, 잘못된 위치에 놓고 성공으로 착각하는 것은 가장 큰 실패다.
- 이동은 가능한 한 `go_to` 중심으로 한다.
- `set_velocity`는 짧은 회전 스캔, source nudge, 복구용으로만 제한적으로 쓴다.
- Level 2의 노하우 중 스캔, 기억, 검증, 장애물 회피 아이디어는 차용하되, Level 1에서 금지되는 `scene_state` 기반 정답 좌표 사용은 하지 않는다.

## 가장 중요한 실패 교훈

- `delivered_count` 증가는 시각적 성공 증거가 아니다.
- 시뮬레이터의 delivered count는 cube가 사라지는 현상도 포함할 수 있으므로, "C 패드에 실제로 놓였는가"를 비판적으로 봐야 한다.
- 특히 green cube는 C pad에 있어야 한다. C에 cube가 없고 conveyor/A 근처에서 사라졌다면 성공이 아니라 큰 실패다.
- `place_entity`를 `{}`로 호출하면 nearest zone에 내려놓을 수 있다. 이 nearest zone이 A/conveyor/source가 되는 사고가 반복됐다.
- 정상 배달 경로에서는 반드시 explicit destination을 넘긴다:
  - red -> `pad_B`
  - green -> `pad_C`
  - blue -> `pad_D`
  - yellow -> `pad_E`
- `place_entity done`이나 SDK `ok=True`만 믿지 않는다.
- 성공 확정은 "matching explicit pad target으로 place했고, held가 비었고, 점수 변화가 맞는 경우"만 허용한다.

## A/Conveyor 규칙

- A는 source/conveyor이며 destination이 아니다.
- A, conveyor, source anchor 주변은 no-place zone이다.
- A 글자판이나 A 방향 conveyor 근처로 place target을 잡으면 안 된다.
- A 근처에서 cube를 집을 수는 있지만, A에 내려놓고 배달 성공으로 처리하면 안 된다.
- 현재 가장 치명적인 버그는 "blue/green cube를 들고도 다시 A를 target으로 잡거나, A에 내려놓고 성공 처리하는 것"이었다.

## 패드 인식 규칙

- B/C/D/E 글자판은 방향 힌트인 landmark다. 곧바로 place 좌표가 아니다.
- 실제 place는 가까운 컨테이너/목재 팔레트 위에서만 한다.
- 색 blob만으로 pad라고 판단하지 않는다.
- candidate는 색 사각형, 글자 흔적, 목재/팔레트 특징, 화면 위치, clipping 여부를 같이 본다.
- 화면 상단이나 가장자리에 잘린 큰 표지판은 drop zone이 아니라 landmark일 가능성이 크다.
- D의 큰 파란 표지판은 특히 drop target이 아니다. D pallet close view에서 다시 검증해야 한다.
- B/C/D/E가 멀리 보이면 그 방향으로 접근하되, 멀리서 바로 place하지 않는다.
- close view에서 `NearPalletCertificate`가 없으면 `place_entity`를 호출하지 않는다.

## 로봇 시야와 스캔

- 기본 head pitch는 약간 아래로 둔다.
- pick/place 직전에는 더 아래를 잠깐 본다.
- 패드가 안 보이면 좌/정면/우를 훑고, 그래도 없으면 분할 회전으로 더 넓게 탐색한다.
- 한 번에 360도 돌기보다 여러 yaw로 나눠서 관찰한다.
- 이동 중에도 너무 자주 보지는 말되, waypoint 후 또는 장거리 이동 후에는 재관찰한다.
- Chrome 외부 화면과 robot POV frame을 모두 봐야 한다. 로그 숫자만으로 성공 판정하지 않는다.

## 지도와 기억

- 정확한 전역 지도보다 관찰 기반 기억이 중요하다.
- 기억해야 할 것:
  - `known_source_xy`
  - `known_pad_xy`
  - `confirmed_pad_xy`
  - `rejected_pad_xy`
  - `no_place_zones`
  - `hazard_zones`
  - 최근 failure events
- unavailable robot status일 때 좌표를 저장하지 않는다.
- rejected/no-place/hazard 근처를 다시 target으로 잡지 않는다.
- confirmed pad는 다음 배달부터 빠른 경로에 사용할 수 있다.
- 단, confirmed pad도 matching color에만 사용한다.

## 내비게이션 규칙

- pad로 갈 때 미확정 좌표로 긴 직선 `go_to`를 하지 않는다.
- 먼 landmark가 보이면 safe approach point나 bounded step으로 나눠 접근한다.
- `go_to` timeout이 나도 robot status가 목표 반경이면 성공 보정할 수 있다.
- 반대로 SDK `done`이어도 실제 좌표가 멀면 place 준비 상태로 보지 않는다.
- 장애물, 선반 내부, source/no-place zone으로 target을 잡지 않는다.
- 벽이나 컨테이너에 정면 돌진을 반복하면 즉시 target을 reject하거나 waypoint를 바꿔야 한다.
- fallen 또는 robot z 비정상이면 반복 행동을 멈추고 recover/stop으로 빠진다.

## Source Farming

- 처음 검증된 A/source 위치를 `source_anchor`로 기억한다.
- source anchor 근처에서는 색 blob을 계속 쫓지 말고 generic `pick_entity(cube)`를 시도한다.
- pick 실패 후 같은 위치에서 무한 반복하지 않는다.
- 짧은 후진/측면 nudge, head down scan, cooldown 후 다시 시도한다.
- 들고 난 뒤 실제 held color가 의도 색과 다르면 held color를 기준으로 route를 다시 잡는다.
- 원하지 않는 색이어도 버리지 말고 해당 색 pad로 배달한다.

## LLM/VLM/API 운용

- LLM은 매 action마다 부르지 않는다. local policy가 실시간 제어를 맡는다.
- LLM은 실패 복구, 우선순위 advisory처럼 느린 판단에만 제한적으로 쓴다.
- Tokamak env가 있으면 OpenRouter보다 우선 사용한다.
- 모델 기본값은 `qwen/qwen3.6-35b-a3b`이다.
- API key는 로그나 문서에 남기지 않는다.
- VLM 호출 지연을 줄이기 위해 이미지 전송 전 해상도를 줄인다.
- 현재 기본 VLM downscale은 긴 변 512px, JPEG quality 65 수준이다.

## 테스트와 검증 루프

- 코드 수정 후 최소 게이트:
  - `python -B -m py_compile menlo_runner\programs\project\ko\level_1_starter_ko.py tests\test_level_1_fast_policy.py`
  - `python -m unittest tests.test_level_1_fast_policy -v`
- Chrome live loop에서 확인할 것:
  - A/conveyor에 place하지 않는가
  - B/C/D/E matching pad를 향해 실제 이동하는가
  - close pallet verify 후에만 place하는가
  - `place_entity`가 explicit `pad_B/C/D/E` target으로 호출되는가
  - 점수 증가 없는 place를 성공으로 기록하지 않는가
  - fallen이 없는가
  - 같은 rejected target을 반복하지 않는가
- 실패 하나를 찾으면 코드와 테스트에 고정하고 다시 live run한다.

## 현재 구현상 핵심 파일

- `menlo_runner/programs/project/ko/level_1_starter_ko.py`
  - Level 1 behavior tree, perception, navigation, explicit placement, memory update.
- `tests/test_level_1_fast_policy.py`
  - A/no-place, pad classifier, explicit place, score verification, navigation supervisor 회귀 테스트.
- `AGENTS.md`
  - 작업 규칙과 검증 절차.

## 절대 다시 하면 안 되는 판단

- "cube가 사라졌으니 성공"이라고 단정하지 않는다.
- "SDK가 done이니 성공"이라고 단정하지 않는다.
- "green 표지판이 보이니 지금 놓아도 된다"라고 판단하지 않는다.
- "A가 가까우니 A에 내려놓자"는 배달이 아니다.
- "멀리 보이는 D 큰 파란 글자판"을 D pallet로 착각하지 않는다.
- Chrome 화면을 보지 않고 로그 숫자만으로 성공 보고하지 않는다.
