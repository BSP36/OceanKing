# OceanKing demo: Claude collects trash on the seabed

**Status (2026-11-01): runs end to end with the real Claude API; not yet reliable.** In the final
configuration, 2 of 8 runs delivered both bottles (9 of 16 bottles in all); about half of the grasps
miss. The run shown in the README delivered both in 55 s of simulated time, retrying one missed
grasp.

```bash
cp .env.sample .env          # fill in CLAUDE_API_KEY (or export ANTHROPIC_API_KEY)
MUJOCO_GL=egl .venv/bin/python -m oceanking.demo.agent --video run.mp4 --transcript run.json
# no API: a scripted stand-in fetches one piece
MUJOCO_GL=egl .venv/bin/python -m oceanking.demo.agent --dry-run --fetch red_bottle --video dry.mp4
```

Options: `--layout showcase` (default: two bottles in front, the zone a turn to the left) or `full`
(three pieces around the robot); `--effort` (default `medium`); `--max-turns`; `--budget-usd` (stops
the run when its estimated spend passes it, default 5; every run is appended to
`scratch/api_ledger.jsonl`); `--no-step-off` (see below).

## What runs

| Layer | File | What it is |
|---|---|---|
| world | `oceanking/demo/scene.py` | `ocean.five_foot` (beta 0.8, still water) with sunken trash and a drop zone (magenta disk, 0.18 m in radius, pole and flag). The bottles have a wide base (7 cm) and a 3.6 cm neck the claw closes on, 2500 kg/m^3 (Archimedes' buoyancy from `oceanking/props.py`); the necks have a rubber grip (below) |
| robot | `oceanking/demo/sim.py` (`OceanSim`) | one environment, native MuJoCo on CPU, the arm-out five-foot policy walking; the carried arm and the claw driven by joint targets; the **onboard camera** (`jumper.xml`'s `onboard`, 95.7 deg vertical field of view) rendered at 640 x 480 |
| perception | `oceanking/demo/perception.py` | colour thresholds calibrated in the water's light, connected components, ground-plane projection of each blob's lowest pixel with the camera's own pose; the robot's own body masked out (it is red); a fix is flagged untrusted when the blob is cut by the image border or its foot is hidden behind the claw. A flat disk's lowest pixel is its near rim, so the zone's fix is moved a radius along the line of sight to its centre |
| skills | `oceanking/demo/skills.py` | `look`, `turn`, `walk`, `approach`, `grasp`, `release`, `go_to`, `status`: odometry-closed moves at the speeds the policy walks, the claw's state from the finger encoder |
| agent | `oceanking/demo/agent.py` | a manual Claude tool-use loop: `claude-opus-5-5`, adaptive thinking, the server-side refusal fallback (`fallbacks: "default"`), prompt caching; `look` returns the onboard image (PNG) with the detector's list; a failing tool is reported back as an error result. The prompt and the tools name only the trash the scene holds |

Claude gets nothing the robot would not have: the onboard image, the detector's output, odometry,
the claw encoder. The positions of the trash are read only by the scorer (`agent.score`).

## The policy: walking with the claw out

The official five-foot policy carries the arm **stowed** and cannot walk with it out: 1.5 s of a
0.15 m/s command walks 0 mm with the arm at the grasp preset. With it stowed, the claw cannot take
anything: swinging it out sweeps the trash aside (`docs/GRASP.md`). So a policy was fine-tuned to
walk with the arm held out (`ocean.five_foot --arm reach_lift`: the arm at the grasp preset, the
shoulder up to 0.5 rad raised, from the official checkpoint, 4096 environments, beta 0.8). It walks
about 0.17 m/s with a piece lifted ~3 cm (`oceanking/tasks/ocean/five_foot/out/reach/model.pt`).

## How a pick-up goes

1. `look` (and `turn`): find the drop zone first, then the target (within 1-3 cm in plain view at
   ~1 m).
2. `approach`: face along the line of the claw's mouth, walk at full speed tracking the target while
   it is farther than 0.45 m, then forward with a sideways trim to 6 cm short, forward pulses
   measured for this policy, and the last sideways centimetres by the arm
   (`oceanking/demo/arm_ik.py`, ±3 cm).
3. `grasp`: close, lift 0.45 rad; held if the finger stopped short (about -0.37 rad on the neck
   against +0.10 shut).
4. `go_to` the drop zone: with a load, **sideways first, then forward or back, never turning**, each
   piece to a spot of its own (the first beyond the zone's centre as seen from the robot, the next
   short of it).
5. `release`. Tool results warn when the claw has closed on nothing on the way, and `look` marks a
   piece lying in the zone as delivered.
6. After `finish`, for the camera: the open claw is raised and the robot steps 0.3 m back, so that
   what it delivered can be seen (`--no-step-off` to skip; the score is taken after it).

## What it took (measured, 2026-10-28 to 11-01)

The first three real runs delivered nothing. Each failure was isolated with the scripted stand-in
and fixed in turn:

| Seen | Cause and fix |
|---|---|
| a held bottle fell out 1-3 s into the first turn | the finger's torque is in the policy's observation and it only ever trained with an empty claw: a squeezed piece saturates the servo far outside that, and the gait comes apart (as upstream measured). The policy is shown the empty claw's value (`sim.FINGER_TORQUE_SEEN`); the grip keeps its full force |
| it slipped out once walking | in a two-jaw pinch the bottle pivots about the pinch axis and only torsional friction holds it; the props' value is a point contact. The necks got a rubber grip (`scene.NECK_FRICTION`: 1.0, torsional 0.01 m) |
| turns with a load still lost it (2 of 3) | `go_to` with a load walks sideways and straight only; scripted, 5 of 5 grasped pieces were delivered (before: 0 of 6) |
| pieces released on the zone's rim | the zone's fix was its near rim (up to 0.18 m short); now its centre (within 2 cm) |
| Claude picked a delivered piece up again and pushed it out | detections in the zone are reported as delivered |
| a full turn searching for a carton the scene does not have | the prompt and the tools' labels name only the scene's trash; finish as soon as all of it is in |
| the second delivery knocked the first out (4 of 6) | a release spot per piece |
| stepping back right after each release dragged the bottle out | reverted: the claw is raised first, and only once, after `finish` |

Real runs by stage (bottles in the zone at the end, by the simulator):

| Stage | Runs | Bottles | Both |
|---|---|---|---|
| before the fixes | 3 | 0 / 6 | 0 |
| grip and carry fixed | 3 | 2 / 6 | 0 |
| zone fix at its centre | 3 | 4 / 6 | 1 |
| the scene's own trash list | 6 | 5 / 12 | 0 |
| a release spot per piece | 6 | 6 / 12 | 2 |
| a step back after each release (reverted) | 6 | 3 / 12 | 0 |
| final (step off after `finish`) | 8 | 9 / 16 | 2 |

Over all 35 runs, 88 of 184 grasps held. A run costs 0.04-0.25 USD.

## Known gaps

1. **Missed grasps**, about half: the bottle is pushed or tipped in the last centimetres. The end of
   the approach is hand-written control; learning it is the next step.
2. **Delivered pieces get knocked** by later moves near the zone.
3. **Odometry is the simulator's base pose**, not an estimate; on a robot it would drift.
4. The rubber neck and the hidden finger torque are choices made for this demo; the second should
   become training with a load in the claw.

Tests (`tests/test_demo_agent.py`, no API): the detector finds the bottle from the spawn and does
not report the robot's own red body; `turn` turns; the loop sends the model, tools, thinking and
refusal-fallback parameters, puts the onboard image into the `look` result, and reports a failing
tool back instead of stopping; the spend cap and the key loading.
