# OceanKing training specification (draft)

**Status: agreed, 2026-10-25; not yet implemented.** Decisions taken: the actor sees the
current (option A, single run); training on `seabed_rock` only, sand in evaluation only.

The physics this trains against is [`PHYSICS.md`](PHYSICS.md).

## 1. Goal

**A crab that walks at the commanded velocity whatever the current and buoyancy**, as far as the
physics allows, and holds its position when commanded to stand.

The official `jumper.posture` policy takes a velocity command and tracks it. Measured on land
(2026-10-25, `--nominal`, 3 s per step): forward 0.2–0.8 m/s at 83–95% of the command, sideways
0.2–0.6 m/s at 80–94%, turning 1–3 rad/s at 90–93% (0.1 m/s forward: 63%). It was trained through
its whole command curriculum (checkpoint: command level 5 of 5, the precision rung at the 0.8 m/s
and 4 rad/s ceilings). Underwater (PHYSICS.md §11) it walks slowly in still water and, in a
0.2–0.3 m/s current, drifts downstream at 0.06–0.10 m/s even when commanded to stand.

The commanded velocity is over the ground, in the body frame (upstream's definition), so a crab
walking at 0.3 m/s against a 0.3 m/s current moves through the water at 0.6 m/s.

### What the physics allows

Traction against drag: the feet hold while `0.5 rho Cd A v_rel^2 < mu (1 - beta) M g`, with
`v_rel` the speed relative to the water (walking speed plus head current) and `Cd A = 0.0818 m^2`
head-on (PHYSICS.md §6.2; side-on is the same within 1%). Limit on `v_rel`, m/s:

| beta | mu 0.4 | mu 0.5 (wet rock) | mu 0.6 (sand) | mu 0.8 | mu 1.2 |
|---|---|---|---|---|---|
| 0.70 | 0.43 | 0.48 | 0.52 | 0.60 | 0.74 |
| 0.75 | 0.38 | 0.42 | 0.46 | 0.53 | 0.65 |
| 0.80 | 0.32 | 0.36 | 0.40 | 0.46 | 0.56 |
| 0.85 | 0.27 | 0.31 | 0.33 | 0.39 | 0.47 |
| 0.90 | 0.22 | 0.24 | 0.27 | 0.31 | 0.38 |
| 0.95 | 0.15 | 0.17 | 0.18 | 0.21 | 0.26 |

A static estimate (all of the net weight on the feet); a gait that lifts feet carries the same net
weight on fewer of them, so the total friction is the same. Evaluation reports, for each cell,
whether the command was within this limit, and "at the commanded velocity" is judged only there.

## 2. Starting point

Fine-tune from the official `jumper.posture` checkpoint (`model_74800.pt`): same network, same
action space, same PPO settings (`jumper.posture`'s `rl_cfg`), with the observation extended
(section 3). The new input columns of the first layer are **initialised to zero** and the
observation normaliser's statistics for them to (mean 0, variance 1), so at iteration 0 the policy
is exactly the official one; training only adds what the water needs.

## 3. Observation

**Actor: upstream's nine terms, unchanged, plus two.** Upstream (`jumper.posture`, 415 values):
`base_ang_vel` (3), `projected_gravity` (3), `joint_pos`, `joint_vel`, `actions` and
`actuator_force` (5-frame strided histories, 100 each), `command` (3), `gait_phase` (2),
`posture_command` (4); noise on the IMU terms as upstream. Added (418 in total):

| Term | Dim | Meaning | Noise | Realism |
|---|---|---|---|---|
| `buoyancy_ratio` | 1 | the robot's current beta | none | known on a real robot: the keel is fixed and the crown tank's fill is the robot's own state |
| `current_body` | 2 | water velocity at the robot, in its yaw (heading) frame | uniform ±0.03 m/s | an onboard flow sensor (decided 2026-10-25; stated in the report) |

**Critic: upstream's fifteen terms** (including the true base linear velocity, foot contacts and
contact forces), **plus** `buoyancy_ratio`, `current_body` without noise, and the ground's friction.

## 4. Randomisation (per environment, resampled at every reset)

Ranges chosen so that **standing is always within the traction limit** and walking at the command
is possible in most of the space (section 1): at the worst corner, beta 0.9 on friction 0.4, the
feet hold up to 0.22 m/s relative to the water.

| Quantity | Range | How |
|---|---|---|
| beta | 0.75 – 0.90 | keel mass per environment (`body_mass`, `set_const`); keel position stays at the beta-0.8 trim, so pitch trim varies slightly with beta, as a real fixed-keel vehicle's would |
| current | speed 0 – 0.2 m/s, uniform direction, constant within an episode | `opt.wind` expanded to one entry per world (probe 2026-10-25: per-world currents work on MuJoCo Warp, and in-place edits take effect without re-capturing the CUDA graph) |
| ground friction | 0.4 – 0.8 | on the ground geom, not the feet (the ground outranks the feet: PHYSICS.md §8) |
| upstream base COM offset | upstream's ranges x 0.5 | upstream's land ranges weigh about twice as much against the underwater net weight |
| upstream encoder bias, observation noise | unchanged | |

**Current curriculum:** the speed range opens from 0–0.05 m/s to 0–0.2 m/s over the first
1000 iterations, or by tracking performance (upstream's curriculum pattern).

Evaluation goes beyond these ranges (beta 0.95, currents 0.3 m/s) to show where the policy, and the
physics, give out.

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
| beta | 0.8, 0.9 (and 0.95, outside training) |
| current | 0, 0.1, 0.2, 0.3 m/s; head-on, cross, following |
| commands | forward 0.2 and 0.4 m/s; sideways 0.2 m/s; turn 1 rad/s; stand |
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
