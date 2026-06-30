# AGENTS.md

## Mission

Build the Level 2 Menlo robot agent slowly and deterministically. Do not jump straight
to live simulator runs. Every change must preserve the Level 2 constraints:

- Do not use `scene_state`.
- Do not use exact cube/pad entity IDs.
- Do not use `go_to`.
- Use camera observation, `set_head`, `set_velocity`, `pick_entity`, `place_entity`,
  `robot_status`, memory, and bounded recovery.

The goal is not to make the robot appear busy. The goal is to move to a cube, pick it,
move to the correct destination pad, place it, verify the result, then repeat until the
task limit is complete.

## Working Rule

Work from 1 to 100:

1. Stop old live processes before changing behavior.
2. Read the relevant code before editing.
3. Make one small conceptual change at a time.
4. Add or update tests for that concept.
5. Run local syntax/tests.
6. Only then run a live simulator test.
7. During live tests, watch both logs and Chrome.
8. If Chrome automation hangs, stop browser automation and use logs/local checks first.
9. Never continue repeated live attempts without writing down the observed failure.
10. Never count a delivery unless the robot was holding an object before place and is
    not holding one after place.

## Simulator Calibration Rule

Level 2 success depends on simulator affordances as much as robotics logic. Treat these
as measured facts, not guesses:

- Record the observed bbox, centroid, blob area, angle, action result, and held-color
  estimate for every live failure.
- Convert each observed simulator failure into a small unit test before another live
  attempt.
- Tune constants from evidence: pickup readiness area/angle, pad placement readiness,
  carried-cube appearance, floor strips, overhead signs, and large edge scene blobs.
- `pick_entity` may pick the nearest physically reachable cube, not necessarily the
  color that the planner intended. Always verify the held color after pick.
- `place_entity` may report success/failure based on simulator proximity. Count a
  delivery only after the robot was holding before place and is not holding after.
- A target that is visually plausible but shaped like signage, a floor strip, or a
  large scene edge must not unlock manipulation.
- If a calibration changes behavior, add or update tests with the exact live bbox/area
  values that motivated it.

## Chrome Control SOP

Chrome is only for visual verification, not for solving robot control logic.

1. Start the robot process and wait for the viewer URL in the stdout log.
2. Open the viewer URL in Chrome.
3. Confirm the log reaches `Skills found`.
4. Capture one screenshot only after the scene is visibly loaded.
5. If Chrome extension control hangs on `goto`, do not keep retrying blindly.
   Use OS-level Chrome open once, then verify via logs.
6. If Chrome still does not join, stop the live robot process and diagnose separately.
7. Do not run multiple live robot processes at once.

## Robot Agent Architecture

The Level 2 agent should be treated as five cooperating modules:

### 1. Observer

Inputs:

- POV camera frame
- `robot_status`
- memory

Outputs:

- visible color detections
- held/not-held state
- low-confidence notes

Rules:

- A color blob alone does not identify a cube or a pad.
- A blob from the carried cube must be filtered out during pad search.
- Observation should be cheap and frequent.
- Any head scan or navigation observation that sees a plausible pad/sign should update
  the pad-bearing memory for that color.

### 2. Planner

Inputs:

- task text
- observation
- memory
- last action result

Outputs:

- one high-level action from the allowed Level 2 action set

Rules:

- Normal cadence must be local and fast.
- LLM may parse task variants or help recovery, but must not block every action.
- The planner must enforce the sequence:
  `search_cube -> navigate_to_cube -> pick_cube -> search_pad -> navigate_to_pad -> place_cube`.

### 3. Navigator

Inputs:

- target kind: `cube` or `pad`
- target color
- current detections

Outputs:

- reached true/false

Rules:

- `navigate_to_cube` may follow cube-colored blobs.
- `navigate_to_pad` must not blindly follow the carried cube.
- Navigation loops must be bounded.
- A failed navigation must return quickly enough for recovery to decide next steps.
- Pad search order is explicit: use remembered pad bearing if available; otherwise
  scan left/front/right, rotate roughly 180 degrees and scan left/front/right again,
  then perform small exploratory moves with rescans.
- Exploration must still avoid Level 2 forbidden helpers: no `scene_state`, no exact
  entity IDs, no `go_to`, and no hard-coded coordinates.

### 4. Manipulator

Inputs:

- validated action
- memory readiness flags

Outputs:

- pick/place result summary

Rules:

- `pick_entity` is allowed only after successful cube navigation.
- `place_entity` is allowed only after successful pad navigation.
- Direct `pick -> place` loops are invalid.

### 5. Verifier and Memory

Inputs:

- action result
- `robot_status`
- optional post-action POV observation

Outputs:

- updated memory
- recent outcome log

Rules:

- Pick success means held state became true.
- Place success means held state became false after a valid held object existed.
- Do not increment delivery count on `place_entity` errors.
- Track repeated failures by target color and action kind.

## Current Known Failure Modes

- Tokamak text model can return only `reasoning_content` with `finish=length`; this
  must not block normal action cadence.
- Tokamak VLM can also return no `content`; VLM hints are off by default and should
  only be enabled with `MENLO_USE_VLM_HINTS=1`.
- Carried cube blobs can look like large target-pad blobs.
- Large pad/sign/scene blobs can be misclassified as cube candidates.
- Wide floor strips, overhead signs, and large screen-edge blobs can look like cubes
  unless cube detection filters use simulator-specific shape constraints.
- `pick_entity` can grab a nearby cube of a different color if the visual target was
  not a real reachable cube; held-color correction is required.
- A carried cube can appear as a small centered low blob, not only as a large blob, and
  must be filtered out during pad navigation.
- Chrome extension control may hang on the Menlo viewer URL; use logs as the source
  of truth for join and action cadence when that happens.

## Test Gate

Before live runs:

```powershell
python -B -m py_compile menlo_runner\programs\project\ko\level_2_starter_ko.py
python -m unittest discover -s tests -v
git -c safe.directory=C:/Users/YEHYUN/Documents/GitHub/hansung-menlo-robotics-workshop diff --check -- menlo_runner/programs/project/ko/level_2_starter_ko.py tests/test_level_2_scenarios.py
```

Live run gate:

- There must be no existing Python robot process from an older run.
- The latest code must have passed the local test gate.
- The live log must be saved under `outputs/level2_liveN_stdout.log` and
  `outputs/level2_liveN_stderr.log`.
- Stop after one live failure and write down the cause before changing code.
