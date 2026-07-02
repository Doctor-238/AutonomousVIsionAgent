# AGENTS.md

## Mission

Build and verify the Level 1 Menlo robot agent for maximum scored deliveries in
the 10 minute benchmark.

Level 1 rules:

- Do not use `scene_state`.
- Do not use exact cube or pad entity IDs.
- Do not use entity-target `go_to`.
- Coordinate `go_to` is allowed only for coordinates estimated from camera
  observations or coordinates recorded after the robot physically reached them.
- Camera observations, `robot_status`, `set_head`, `set_velocity`,
  coordinate `go_to`, `pick_entity`, `place_entity`, memory, and progress helpers
  are allowed.

The scoring target is not "looks busy". The robot must pick a cube from the
source conveyor area, identify the held color, move to the matching destination
pad, place it, verify that the score increased, and repeat quickly.

## Working Rule

Work from evidence, one failure at a time:

1. Stop old live robot processes before a live run.
2. Read the relevant code and latest logs before editing behavior.
3. Make each conceptual change testable.
4. Add or update tests for the behavior being changed.
5. Run the local syntax and unit gates before live testing.
6. During live tests, watch both logs and Chrome.
7. Save robot POV frames and Chrome screenshots when diagnosing behavior.
8. Do not repeat live runs without naming the observed failure mode.
9. Never count a delivery unless `place_entity` happened while holding a cube and
   the delivered-count helper increased afterward.
10. Keep unrelated dirty files untouched.

## Level 1 Architecture

Treat the agent as a behavior tree:

```text
warmup -> map_source -> pick_from_source -> classify_held
       -> locate_pad -> navigate_pad -> place_verify -> return_source
```

The LLM is an advisory planner, not the real-time controller. The local policy
must make fast decisions every cycle. LLM calls may help with recovery and
priority updates, but they must be bounded by a short timeout and must not block
normal action cadence.

## Source Farming

The source conveyor is the production station.

- First find a reachable cube/source cluster from camera observations and record
  the robot position after a successful pick as `source_anchor`.
- Once the anchor is known, return to it instead of chasing every colored blob.
- Near the anchor, prefer direct generic `pick_entity(cube)` because the simulator
  can pick the nearest reachable conveyor cube.
- If a direct source pick fails, do not repeat forever. Back up or sidestep
  briefly, lower the head, rescan, then retry.
- The planned color is only a hint. The held color returned by the progress
  helper is authoritative; deliver whatever color was actually picked.

## Smart Pad Classifier

A destination pad is not just a colored blob.

- A valid pad candidate should look like a colored sign/target with white-letter
  evidence or nearby wood/pallet evidence.
- The A sign/conveyor area is not a destination.
- Use fixed sign semantics only as interpretation:
  B/red, C/green, D/blue, E/yellow.
- Store `known_pad_xy` only from robot-status + camera-derived estimates.
- Store `confirmed_pad_xy` only after a scored placement.
- Store `rejected_pad_xy` after failed navigation or a placement that did not
  increase the score.
- If `robot_status` is unavailable, do not create or update coordinate memory.

## Topological Memory

Use a measured, partial map rather than fixed answer coordinates.

- Track `source_anchor`, `known_pad_xy`, `confirmed_pad_xy`, `rejected_pad_xy`,
  recent failed target positions, and delivery outcomes.
- Opportunistically remember other pad candidates seen during scans.
- Search order for missing pads: left/front/right scan, rotate about 180 degrees,
  left/front/right scan again, then perform small exploratory moves with rescans.
- Rejected coordinates should decay only after better evidence, not immediately.

## Navigation Supervisor

Coordinate navigation is allowed, but it must be supervised.

- Before `go_to`, validate that the target coordinate came from observation or
  recorded memory and is not near a rejected coordinate.
- Do not navigate on fake fallback `(0, 0)` status.
- If `go_to` times out, read `robot_status`; if the robot is already within the
  target tolerance, treat it as reached.
- Near a target, use only short, conservative `set_velocity` nudges. Do not drive
  straight into walls, shelves, or container boards.
- Keep the default head pitch slightly downward. Use a lower close-look pitch
  immediately before pick or place.
- Stop repeated actions if the robot is fallen.

## Visual Forensics

Live testing must produce enough evidence to debug like a video.

- Save annotated POV frames during scans and after actions when diagnostics are
  enabled.
- Capture Chrome viewer screenshots during live smoke tests.
- Log for each action: duration, robot xy/yaw, held color, target kind, target xy,
  bbox, centroid, area, bearing, pad letter score, pad wood score, raw delivered
  count, and scored delta.
- Classify failures as one of:
  `rpc_ready`, `source_pick`, `wrong_pad`, `navigation`, `place_verify`, `fallen`.
- Convert each repeated failure into a unit test or calibration fixture before
  another live attempt.

## Simulator Calibration

Treat simulator affordances as measured facts.

- Tune pickup radius, source revisit radius, placement tolerance, pad feature
  thresholds, target bbox ranges, and `go_to` arrival tolerance from logs.
- Keep tuning values as constants or environment variables.
- `set_sim_speed` is off by default for scored runs. Use it only behind an
  explicit experiment flag.

## Test Gate

Before live runs:

```powershell
python -B -m py_compile menlo_runner\programs\project\ko\level_1_starter_ko.py tests\test_level_1_fast_policy.py
python -m unittest tests.test_level_1_fast_policy -v
```

Live run gate:

- No old Python robot process is running.
- The viewer URL is written to `run_logs/latest_level1_url.txt`.
- Chrome opens that exact URL from the file, not manual copy/paste.
- Logs and screenshots are saved under `run_logs/`.
- Stop after a clear live failure, write down the cause, then patch and retest.
