# OceanKing training specification (draft)

**Status: agreed, 2026-10-25; not yet implemented.** Decisions taken: the actor sees the
current (option A, single run); training on `seabed_rock` only, sand in evaluation only.

The physics this trains against is [`PHYSICS.md`](PHYSICS.md).

## 1. Goal

One policy, `ocean.posture`'s interface, that walks on the seabed under a range of buoyancy
ratios and currents, and **does not get carried away**: it tracks the commanded velocity where the
physics allows it, and holds its position when commanded to stand.

The baseline it must beat is the official `jumper.posture` policy, which underwater
(PHYSICS.md §11) walks slowly in still water and, in a 0.2–0.3 m/s current, drifts downstream at
0.06–0.10 m/s even when commanded to stand — while standing on six feet would hold.

What "possible" means is set by traction against drag: at beta 0.8, head-on, the relative water
speed the feet can resist is about 0.56 m/s at friction 1.2, 0.36 on wet rock and 0.40 on sand
(PHYSICS.md §8, §11). The policy is not asked to beat that limit, and evaluation reports it.

## 2. Starting point

Fine-tune from the official `jumper.posture` checkpoint (`model_74800.pt`): same network, same
action space, same PPO settings (`jumper.posture`'s `rl_cfg`), with the observation extended
(section 3). The new input columns of the first layer are **initialised to zero** and the
observation normaliser's statistics for them to (mean 0, variance 1), so at iteration 0 the policy
is exactly the official one; training only adds what the water needs.

## 3. Observation

The actor keeps every upstream term (IMU, joint history, commands, gait phase, posture command).
Added:

| Term | Dim | Meaning | Realism |
|---|---|---|---|
| `buoyancy_ratio` | 1 | the robot's current beta | known on a real robot: the keel is fixed and the crown tank's fill is the robot's own state |
| `current_body` | 2 | water velocity at the robot, in the body's yaw frame | needs a flow sensor (stated in the report) |

The critic additionally gets the ground friction and the true current (privileged).

**Decided (2026-10-25): the actor sees the current** (option A). The alternative, a second run
whose actor sees only beta, was set aside for time; it stays a possible follow-up.

## 4. Randomisation (per environment, resampled at every reset)

| Quantity | Range | How |
|---|---|---|
| beta | 0.75 – 0.95 | keel mass per environment (`body_mass`, `set_const`); keel position stays at the beta-0.8 trim, so pitch trim varies slightly with beta, as a real fixed-keel vehicle's would |
| current | speed 0 – 0.3 m/s, uniform direction, constant within an episode | `opt.wind` expanded to one entry per world (probe 2026-10-25: per-world currents work on MuJoCo Warp, and in-place edits take effect without re-capturing the CUDA graph) |
| ground friction | 0.4 – 0.8 | on the ground geom, not the feet (the ground outranks the feet: PHYSICS.md §8) |
| upstream base COM offset | upstream's ranges x 0.5 | upstream's land ranges weigh about twice as much against the underwater net weight |
| upstream encoder bias, observation noise | unchanged | |

**Current curriculum:** the speed range opens from 0–0.1 m/s to 0–0.3 m/s over the first
1000 iterations, or by tracking performance (upstream's curriculum pattern).

## 5. Ground

A run uses one scene. **Train on `seabed_rock`** (uneven, friction randomised over 0.4–0.8,
which spans wet rock 0.5 and sand 0.6). Evaluate on `seabed_rock`, `seabed_sand` and `seabed`.
Sand's compliance and granular drag are then seen only in evaluation — a test of generalisation,
stated as such (decided 2026-10-25).

## 6. Reward

Upstream's rewards and weights, unchanged, except:

| Term | Upstream | Underwater |
|---|---|---|
| `stance_load` threshold | 5 N per foot | scaled per environment by `(1 - beta) M / M_land` (the net weight it can actually press with) |
| `stance_load_stand` threshold | 3 N per foot | same scaling |

Without this the two terms can never saturate underwater and push the policy to press its feet
harder than the water allows (review of 2026-10-25). Command tracking is the incentive to hold
station: a zero command in a current is tracked only by not drifting.

## 7. Commands

Upstream's ranges (up to 0.8 m/s, 4 rad/s), unchanged. Commands beyond what the current allows are
kept: tracking rewards saturate smoothly, and the policy learns to do its best rather than being
shown only feasible tasks. A share of episodes (upstream's standing fraction) commands zero, which
is the station-keeping test.

## 8. Run bookkeeping

Each run writes the resolved water condition, randomisation ranges, scene, drag calibration and
source commit into its log directory and its checkpoints' metadata; replay and evaluation read it
and refuse a mismatch unless overridden. (Missing today: PHYSICS.md §13.)

## 9. Evaluation

Frozen before comparing checkpoints. MuJoCo Warp (patched to match native MuJoCo, PHYSICS.md §6.3),
seeded, the same initial states and command sequences for every policy.

| Axis | Values |
|---|---|
| ground | `seabed`, `seabed_rock`, `seabed_sand` |
| beta | 0.8, 0.9 |
| current | 0, 0.15, 0.3 m/s; head-on, cross, following |
| commands | forward 0.3 m/s; sideways 0.2 m/s; stand |
| episodes | 20 per cell |

Metrics: velocity tracking error, drift while commanded to stand (m/s), falls, the traction limit
for the cell (whether the command was physically possible). Policies: official and adapted. Videos for the report come from the same cells.

## 10. Budget and schedule

Measured: 1.88 s per iteration at 4096 environments on flat ground (PHYSICS.md §11).

| Dates | Work |
|---|---|
| Oct 26 | observation terms, randomisation, reward scaling, bookkeeping, checkpoint surgery; smoke runs |
| Oct 26–27 | training run (target 3000–5000 iterations, 1.5–2.5 h each attempt) |
| Oct 28 | evaluation grid, comparison videos |
| Oct 29–31 | mission demo, crown tank control, rendering |
| Nov 1 | report and video |
