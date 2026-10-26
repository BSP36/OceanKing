# OceanKing training specification (draft)

**Status: implemented 2026-10-25 (task `ocean.walk`), not yet trained.** Decisions taken: the
actor sees the current (option A, single run); training on `seabed_rock` only, sand in evaluation
only; beyond the traction limit the crab may be carried away.

Implementation: [`oceanking/tasks/ocean/walk/`](../oceanking/tasks/ocean/walk) (the task, every
number in its `env_cfg.py`), [`oceanking/water.py`](../oceanking/water.py) (per-environment water,
observations, scaled rewards, metrics), [`oceanking/tools/expand_checkpoint.py`](../oceanking/tools/expand_checkpoint.py)
(starting checkpoint). Tests: [`tests/test_water.py`](../tests/test_water.py).

```bash
.venv/bin/python -m oceanking.tools.expand_checkpoint scratch/official_expanded
cd jumper
../.venv/bin/python scripts/train.py --task ocean.walk --scene seabed_rock \
    --checkpoint ../scratch/official_expanded/model_74800.pt --max-iterations 3000 --headless
```

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

A static estimate (all of the net weight on the feet). **Checked** (2026-10-25): a PD-held standing
robot under a slowly rising head current starts to slide at 0.88–1.10 of it (creep of 1 cm,
which includes the legs' elastic give). **Walking reaches less**: the official policy commanded
0.8 m/s in still water on rock over beta 0.75–0.90 x friction 0.4–0.8 tops out at 0.67 of the
limit forward (0.56–0.75) and 0.63 sideways (0.39–0.73), and its top speed correlates 0.99 with
the limit: walking is traction-bound at about two thirds of the static figure (feet lift, legs
swing through the water). Evaluation reports, for each cell,
whether the command was within this limit, and "at the commanded velocity" is judged only there.

**Beyond the limit the crab is allowed to be carried away** (decided 2026-10-25): no policy can
hold there, and a real crab is swept off by a strong enough flow too. Training stays inside the
limit for standing (section 4); evaluation shows what happens outside it.

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
`posture_command` (4); noise on the IMU terms as upstream. Added (419 in total):

| Term | Dim | Meaning | Noise | Realism |
|---|---|---|---|---|
| `buoyancy_ratio` | 1 | the robot's current beta | none | known on a real robot: the keel is fixed and the crown tank's fill is the robot's own state |
| `current_body` | 3 | water velocity at the robot, in its yaw (heading) frame: forward, left, up (up is 0 here; three components because upstream's symmetry augmentation mirrors 3-vectors) | uniform ±0.03 m/s | an onboard flow sensor (decided 2026-10-25; stated in the report) |

**Critic: upstream's fifteen terms** (602 values, including the true base linear velocity, foot
contacts and contact forces), **plus** `buoyancy_ratio`, `current_body` without noise, and the
ground's friction (607 in total).

Upstream mirrors every sample left-right (symmetry augmentation): beta and friction are scalars,
the current a 3-vector whose left component negates.

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

Upstream's 25 terms and weights, with two changes.

### 6.1 Track what the physics allows (revised 2026-10-25)

Upstream's `track_linear_velocity` pays `exp(-|v_cmd - v|^2 / std^2)`. Under water most
commands of upstream's range (up to 0.8 m/s) cannot be reached: against the traction limit of
section 1 (0.2–0.5 m/s in the training ranges), the term pays almost nothing however the robot
walks, so it stops pulling. **Observed in the first run** (stopped at iteration 75950, after the
current reached 0.2 m/s): the term fell from 0.76 to 0.62 while the posture terms rose, and against
the official policy the run walked slower in every condition (forward 0.17 against 0.23 m/s,
sideways 0.09 against 0.19 m/s, rock, beta 0.8, still water).

The term now tracks the **feasible target** instead of the command
(`oceanking.water.track_feasible_velocity`):

- the feet hold up to the traction limit `v_lim` relative to the water in any direction
  (per environment, from its beta, mass, ground friction and the calibrated drag);
- so the reachable ground velocities are the disk of radius `v_lim` centred on the current, in the
  heading frame;
- the target is the command if it is inside the disk, otherwise the disk's nearest point to it:
  as fast as the physics allows towards the command.

Same form, same `std` (still set by upstream's curriculum). Walking at the physical maximum
towards an impossible command is full marks; drifting at the least possible speed in a current
the feet cannot hold is full marks too.

**Refined (2026-10-25), after measuring what walking reaches (section 1):**

- the limit is scaled by `TRACTION_MARGIN = 0.65`, the measured walking fraction of the static
  limit (forward 0.67, sideways 0.63);
- **one-sided beyond the target**: for a command outside the disk, the velocity relative to the
  water is split along and across the direction towards the command. Across is error; along,
  falling short of the target is error and so is passing the command itself, but anything in
  between is not. A conservative margin therefore never caps a policy that learns to walk faster
  than the official one did.

The second run (iteration 75800, before this refinement, margin 1.0) already walked sideways
1.6x faster than the first and drifted less than the official policy in currents, but walked
forward slower than the official policy (0.17 against 0.23 m/s, rock, beta 0.8, still water):
the target it was chasing, the static limit, is a third above what walking can reach.

**Run 3 (2026-10-26), from the review of runs 1–2.** Run 2 (margin 1.0, two-sided) kept the
official top speed (0.25/0.27 m/s forward/sideways at a 0.8 command, against 0.27/0.26) but lost
command gain (a 0.15 m/s command walked 0.078 against 0.103) and drifted more when standing (0.030
against 0.003 m/s). Five reviewers traced it to the reward and the commands, not to too little
training:

- most commands were out of reach, so the gradient mostly paid for "as fast as possible", and the
  style terms (gait, smoothness, energy) pay for walking slower: the cheapest compromise was a
  uniformly slower gait;
- a margin of 0.65 would have made the official speed the ceiling of what is rewarded, so the
  margin stays **1.0** (the physical limit, an aspiration), with the one-sided form above;
- the tracking kernel (upstream's 0.2 m/s land width) is wide against 0.2–0.5 m/s limits: it now
  scales as `std = 0.4 v_lim`, clamped to [0.05, upstream's std] (`TRACK_STD_FRACTION`);
- a **progress term** (`water_progress`, weight 1.5) pays, for out-of-reach commands only, the
  speed along the command relative to the water as a share of the target, so walking faster still
  pays where the tracking kernel saturates.

### 6.4 PPO for fine-tuning (run 3)

| Setting | Upstream | ocean.walk | Why |
|---|---|---|---|
| `desired_kl` | 0.01 | 0.005 | at 0.01 the actor random-walked ~20% from the official weights |
| action std | learned | fixed at the official values | it rose 0.20 → 0.26, a tax through `action_rate_l2` |
| entropy coefficient | 0.01 | 0 | moot with a fixed std |
| optimiser | restored | fresh Adam, lr 1e-4 | the widened checkpoint's moments gave the new columns several times the normal step |

The actor's current observation carries flow-sensor noise on its horizontal components only (the
vertical one is always zero; noise there was a pure-noise input).

**Run 3 result and run 4 (2026-10-26).** Run 3 (1500 iterations from the official checkpoint)
passed the current, station-keeping and no-fall criteria of section 9 (into head currents 1.1–1.3x
the official speed, standing drift at or below the official's) but not the small-command one:
0.15 m/s commands walked 0.70–0.75x the official speed, declining with training (forward 0.102 at
iteration 75100, 0.065 at 75800), and backward 0.3 m/s about 0.7x everywhere. Two reviewers (reward
accounting replayed per term; gait recording) agreed on the mechanism:

- at the 2 Hz cadence floor of upstream's gait clock, small-command speed is stride length only,
  and run 3 strides 18 mm where the official policy strides 29 mm (forward 0.15), with the same slip
  efficiency;
- the reward prefers it: per second, run 3 earns 0.3–0.4 **more** than the official gait at 0.15
  commands, losing 0.3–0.6 on tracking and gaining more on `tripod_gait` (clean contacts),
  `track_height` (land width 0.01 m), `foot_swing_height` and `stance_load`. With the kernel width
  0.4 v_lim (0.145 m/s at beta 0.8, friction 0.5), walking 0.10 m/s for a 0.15 command cost 0.22/s;
- straight small commands were rare: the in-disk draw kept upstream's yaw rate (uniform ±4 rad/s or
  heading control), so only ~5% of training looked like walking straight at 0.1–0.2 m/s.

Run 4 changes the reward and commands, nothing else:

| Change | Value |
|---|---|
| tracking width for a moving target | `min(0.4 v_lim, max(0.5 |target|, 0.06))` (`TRACK_TARGET_FRACTION`, `TRACK_STD_MIN`) |
| `water_progress` also pays reachable commands ≥ 0.08 m/s: ground speed along the command / its speed, clipped to [0, 1], times staying on its line | `progress_towards_command`, weight 1.5 → 2.0 |
| half of the in-disk commands walk straight (no yaw rate, no heading control); in-disk commands stay body-frame | `STRAIGHT_SHARE = 0.5` |

Considered and left: widening `track_height` (its width is set by upstream's posture curriculum),
lowering `tripod_gait` (it rewards clean contacts, not wrong in itself). Run 4 starts again from the
official checkpoint, and its checkpoints are evaluated every 100 iterations, since the last one need
not be the best.

### 6.2 Foot loads follow the net weight

| Term | Upstream | Underwater |
|---|---|---|
| `stance_load` (+0.5) | stance feet loaded to 5 N each | threshold scaled per environment by its net weight over the land weight, `(1 - beta) M / m_land` (beta 0.8: 2.2 N) |
| `stance_load_stand` (+1.167) | standing feet loaded to 3 N each | same scaling (beta 0.8: 1.3 N) |

Without it the two terms cannot saturate underwater (net weight 10.8 N at beta 0.8 against 25 N
on land) and push the policy to press its feet harder than the water allows.

### 6.3 Unchanged, and why

| Term | |
|---|---|
| `track_angular_velocity` (+2) | turning in place meets little drag; no evidence of it being infeasible |
| `soft_touchdown` (−4.5) | a speed, independent of weight |
| `foot_slip` (−0.5), `standing_sway` (−0.1) | already resist being carried |
| `energy` (−0.02), `torque_headroom` (−1) | servo protection, which resisting a current must respect |
| the rest (gait, posture, clearance, limits, smoothness) | |

**Watched during training**, not rewarded: `feasible_tracking_error` (m/s, against the target),
`command_feasible` (share of steps whose command was reachable as given), drift while
commanded to stand.

## 7. Commands

From run 3, **half of the moving commands** are redrawn uniformly inside the disk the environment
can walk (`WALK_FRACTION = 0.65` of its traction limit, what the official gait reaches;
`oceanking.water.OceanVelocityCommand`), so small and reachable commands, where command gain is
learned, are a real share of training. The rest keep upstream's ranges: (up to 0.8 m/s, 4 rad/s), unchanged: with the feasible target an
out-of-reach command means "as fast as possible that way", which is what a command to a crab in
a current should mean. A share of episodes (upstream's standing fraction) commands zero, the
station-keeping test. After a resume the first command of each environment is still drawn from the
curriculum's first rung (it was sampled before the checkpoint's level was restored) until its
first resample, 3–8 s later.

## 8. Run bookkeeping

Each run writes the resolved water condition, randomisation ranges, scene, drag calibration and
source commit into its log directory and its checkpoints' metadata; replay and evaluation read it
and refuse a mismatch unless overridden. (Missing today: PHYSICS.md §13.)

## 9. Evaluation

**Fixed protocol from run 3** (`oceanking.tools.evaluate`, frozen before run 3 finished): native
MuJoCo on CPU, one environment from the nominal start without observation noise, the water pinned;
the official policy through its expanded checkpoint. Seven walking conditions (rock beta 0.8
friction 0.5 still, head 0.1, head 0.2, cross 0.2; rock beta 0.75 friction 0.8; rock beta 0.9
friction 0.4; sand beta 0.8), each walking forward 0.15/0.3/0.8, sideways 0.15/0.3/0.8 and backward
0.3 m/s for 4 s; three standing runs in currents. A candidate passes only if all hold against the
official policy in the same runs:

| | Criterion |
|---|---|
| A | small commands (0.15 forward and sideways, reference water) ≥ 0.95x |
| B | geometric mean of the speed ratio over all walking segments ≥ 1.0 |
| C | forward speed into the two head currents ≥ 1.05x |
| D | drift when standing ≤ official + 0.005 m/s in every standing run |
| E | no fall (body up-axis stays above 0.5) |

The broader grid below remains the plan for the report.

Original plan, frozen before comparing checkpoints. MuJoCo Warp (patched to match native MuJoCo, PHYSICS.md §6.3),
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

Measured (2026-10-25, 4096 environments, `ocean.walk` on `seabed_rock`, resumed from the expanded
official checkpoint): **3.47 s per iteration**, i.e. about 1 h per 1000 iterations.

What a resumed run starts from (verified): iteration 74800, the command curriculum at level 5 of 5
and the posture curriculum at level 3, as the official run left them; the expanded actor's
actions equal the official actor's exactly whatever the water columns read.

| Dates | Work |
|---|---|
| Oct 26 | observation terms, randomisation, reward scaling, bookkeeping, checkpoint surgery; smoke runs |
| Oct 26–27 | training run (target 3000–5000 iterations, 1.5–2.5 h each attempt) |
| Oct 28 | evaluation grid, comparison videos |
| Oct 29–31 | mission demo, crown tank control, rendering |
| Nov 1 | report and video |
