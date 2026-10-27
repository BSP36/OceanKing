# OceanKing demo: Claude collects trash on the seabed

**Status (2026-10-27): implemented and tested without an API key** (a scripted stand-in drives
the same loop). The robot, its skills and the Claude loop run end to end; the pick-up is not yet
reliable (below).

```bash
# with credentials for the Claude API (ANTHROPIC_API_KEY, or `ant auth login`):
.venv/bin/python -m oceanking.demo.agent --video run.mp4 --transcript run.json
# no API: a scripted stand-in fetches the can
.venv/bin/python -m oceanking.demo.agent --dry-run --video dry.mp4
```

## What runs

| Layer | File | What it is |
|---|---|---|
| world | `oceanking/demo/scene.py` | `ocean.five_foot` (beta 0.8, still water) with three pieces of sunken trash (red can, green bottle, yellow carton; effective densities 1300-1500 kg/m^3, Archimedes' buoyancy from `oceanking/props.py`) and a drop zone (magenta disk, pole and flag) |
| robot | `oceanking/demo/sim.py` (`OceanSim`) | one environment, native MuJoCo on CPU, a five-foot policy walking; the carried arm and the claw driven by joint targets; the **onboard camera** (`jumper.xml`'s `onboard`, 95.7 deg vertical field of view) rendered at 640 x 480 |
| perception | `oceanking/demo/perception.py` | colour thresholds calibrated in the water's light, connected components, ground-plane projection of each blob's lowest pixel with the camera's own pose; the robot's own body masked out (it is red, as the can is); a fix is flagged untrusted when the blob is cut by the image border or its foot is hidden behind the claw |
| skills | `oceanking/demo/skills.py` | `look`, `turn`, `walk`, `approach`, `grasp`, `release`, `go_to`, `status`: odometry-closed moves at the speeds the policy walks, tracking the target with the camera while it is far enough to be seen whole, the claw's state from the finger encoder |
| agent | `oceanking/demo/agent.py` | a manual Claude tool-use loop: `claude-opus-5-5`, adaptive thinking, effort `medium` (flag), the server-side refusal fallback (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`); `look` returns the onboard image (PNG) with the detector's list; a failing tool is reported back as an error result |

Claude gets nothing the robot would not have: the onboard image, the detector's output, odometry,
the claw encoder. The positions of the trash are read only by the scorer (`agent.score`).

## The policy: walking with the claw out

The official five-foot policy carries the arm **stowed** and cannot walk with it out: 1.5 s of a
0.15 m/s command walks 0 mm with the arm at the grasp preset (native, 2026-10-27). With it stowed,
the claw cannot take anything: swinging it out sweeps the trash aside (`tools/grasp_check.py`
variants; docs/GRASP.md). So a policy was fine-tuned to walk with the arm held at the grasp preset
(`ocean.five_foot --arm reach`: the hold and the default pose moved to it, ±0.10 rad per episode;
from the official checkpoint, 4096 environments, beta 0.8). After 200 iterations it walked 0.105 m/s
forward, 0.097 sideways and turned 0.73 rad/s with the arm out, and short forward pulses moved
it 11-55 mm (0.2-0.5 s), which is what the last centimetres of an approach need.

## How a pick-up goes

1. `look`: the detector's fix for the target (within 1-3 cm in plain view at ~1 m).
2. `approach`: walk at full speed, turning to keep the target in view and refreshing the fix while
   it is farther than 0.45 m; then forward with a sideways trim (sideways pulses from standing do
   not move this gait at all) to 6 cm short; forward pulses measured for this policy; the last
   sideways centimetres by the arm (`oceanking/demo/arm_ik.py`, ±3 cm). The target is brought
   1.5 cm inside the mouth: the fix reads ~1.4 cm short at the end.
3. `grasp`: close; lift by the 0.1 rad the policy walks with (a larger lift left it unable to walk);
   held if the finger stopped short (about -0.4 rad on a 40 mm piece against +0.10 shut).
4. `go_to` the drop zone: turn to face it, walk with a trim; a zone seen without its foot is
   walked towards by bearing and looked at again.
5. `release`.

## Where it stands

Measured with the scripted stand-in (fetch the can), checkpoints of the arm-out fine-tune:

- approach and grasp: the claw closed on the can and held it in most runs (0.2-0.7 cm from the
  fix, finger stopped at -0.37 to -0.43 rad); in some runs the claw or a leg knocked the can over on
  the way in;
- carrying: held through a 90 deg turn with the 0.1 rad lift (under 1 cm of slip); in one full run
  it dropped out on the way;
- delivery: `go_to` reached the zone (4 cm from its fix) once the zone was seen.

Not yet a reliable end-to-end pick-and-place; the loop around it is complete. Known gaps, in order of
what they cost:

1. **Knocking the trash over on the way in.** The light trash under water (net weight ~0.3 N) tips
   at a touch. Candidates: approach slower in the last 15 cm; trash that lies flat.
2. **The grip is shallow.** With a 0.1 rad lift the trash only just clears the floor. A policy
   trained to walk with the arm raised (a second hold pose) would carry it clear.
3. **Odometry is the simulator's base pose**, not an estimate; on the robot it would drift.

Tests (`tests/test_demo_agent.py`, no API): the detector finds the bottle within 10 cm from the spawn
and does not report the robot's own red body (control: unmasked it does); `turn` turns; the loop
sends the model, tools, thinking and refusal-fallback parameters, puts the onboard image into the
`look` result, and reports a failing tool back instead of stopping.
