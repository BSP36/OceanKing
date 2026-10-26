# OceanKing jump specification (ocean.jump)

**Status: implemented 2026-10-26, first training run.** Decisions taken with the user: the
command is a **direction and a strength**; beta up to **0.98**; the crown tank stays fixed during
a jump (tank control later, as its own policy).

Implementation: [`oceanking/tasks/ocean/jump/`](../oceanking/tasks/ocean/jump) (environment,
command, rewards, observations; every number in `env_cfg.py`). Starting checkpoint:

```bash
.venv/bin/python -m oceanking.tools.expand_checkpoint --task jump scratch/official_jump_expanded
cd jumper
../.venv/bin/python scripts/train.py --task ocean.jump \
    --checkpoint ../scratch/official_jump_expanded/model_5999.pt --max-iterations 3000 --headless
```

## 1. Goal

**Jump towards a commanded landing point, rising as high as the water allows on the way**: the
picture is a crab that springs up with its legs and the buoyancy, floats, and lands where it
aimed. Straight up when commanded with strength 0.

## 2. What the physics allows (measured 2026-10-26)

The official `jumper.ref_free_jump` policy (`model_5999`), replayed in `ocean.jump`'s water at
beta 0.8 (still water, rock-like plane):

| | land | water, beta 0.8 |
|---|---|---|
| take-off vertical speed | 1.76 m/s | **0.76 m/s** |
| rise of the body | +0.28 m | **+0.15 m** |
| airtime | ~0.45 s | ~1.0 s |

It still jumps, but its push is tuned for land: the legs drive through water (drag, added mass)
and the take-off speed halves. On the way down it tilts ~23°.

Ballistic flight of the robot held at HOME, launched at 2.4 m/s, no ground
(`scratch/jump/ballistic.py`; range to the launch height):

| | range | apex | airtime |
|---|---|---|---|
| land, 45° | 0.59 m | 0.15 m | 0.35 s |
| beta 0.8, 45° | 0.40 m | 0.21 m | 1.1 s |
| beta 0.8, 90° | 0 | 0.34 m | 1.4 s |
| beta 0.95, 45° | 0.46 m | 0.29 m | 2.6 s |
| beta 0.98, 45° | 0.51 m | 0.34 m | 4.5 s |
| beta 0.98, 90° | 0 | 0.50 m | 5.9 s |

- **Height and airtime gain** under water (the net weight is small);
- **horizontal reach does not**: drag takes the horizontal speed in a few tenths of a second, so
  one jump lands 0.4–0.5 m away at most, at a take-off speed the official policy reaches on land
  only. **Full strength is therefore 0.3 m** (`REACH`), reachable at a realistic take-off.

## 3. Task

| | |
|---|---|
| command | strength s in [0, 1] (0 in 20% of episodes: straight up) and direction, uniform; target landing point s × 0.3 m from the take-off point, drawn once per episode |
| beta | per environment at startup, uniform 0.75–0.98 (keel mass) |
| ground friction | per reset, uniform 0.4–0.8 |
| current | none (a flight of seconds at high beta would be carried 0.1–0.5 m by 0.1 m/s: later) |
| actor observation | upstream's 406 + the target in the heading frame, in units of the reach (3: forward, left, 0) + beta = **410** |
| critic observation | upstream's 435 + target, beta, ground friction, offset of the target from the robot (m, heading frame) = **443** |
| episode | 8 s (upstream 2.5 s: a flight at beta 0.98 lasts up to ~6 s); ends early once back at HOME |

The command extends upstream's `RefFreeJumpCommand` (one jump per episode, its own flight
detected from the feet's contacts, apex latched, second jumps clawed back) with the target and the
landing point. Its `jump_done` excludes flight: upstream's ended the jump once the recording's span
ran out, which under water is mid-flight, and `back_home` would end the episode in the air.

## 4. Reward

Upstream's terms and weights, with these changes (as of run 2):

| Term | Upstream | ocean.jump |
|---|---|---|
| `landing_target` (new, +6) | | `exp(-|landing - target|^2 / 0.08^2)`, paid every step after the episode's own jump has landed, nothing after a second jump |
| `landing_near` (new, +4) | | the same with a 0.25 m kernel: still pays a full-strength jump that lands 0.3 m off (0.24) |
| `apex_height` (+4) | rise of the lowest point / 0.1 m, after landing | the same **times the wide landing accuracy**: height counts on a jump that lands roughly where aimed |
| `flight_height` (+2) | rise during flight | **0**: it pays every airborne step, and at beta 0.98 a flight lasts seconds, which would pay more than any landing |
| `horizontal_drift` (−1) | distance from the start | `anchored_drift`: beyond 0.05 m from the take-off point before the jump, nothing in flight, from the landing point after it (no shuffling towards the target, no sliding after landing) |

**Run 1 (stopped at iteration 6650 of 9000)** had only the narrow kernel, scaled the apex by it,
and charged all travel before take-off. At its iteration 6500 the policy landed within 0.06 m of
its take-off point whatever the target (mean error 0.25 m over aimed targets, the official 0.27 m)
and jumped lower (rise 0.12 m at beta 0.8, the official 0.16 m): a missed full-strength target paid
`exp(-14)`, nothing to climb, while not aiming kept the drift penalty and the apex scaling quiet.

## 5. Training

Fine-tuned from the official checkpoint, iteration 5999, widened with zero columns
(`expand_checkpoint --task jump`): it acts exactly as the official policy, whatever the new columns
read. At that iteration upstream's training aid (the recording fed forward, and the prior on the
joints) is already gone, and the environment's step counter is restored with it.

PPO as ocean.walk's fine-tuning (docs/TRAINING.md section 6.4): fresh Adam at lr 1e-4, the action
std fixed at the official values, no entropy bonus; `desired_kl` stays at upstream's 0.01, since an
aimed jump is a new skill rather than a correction. 4096 environments: 1.2 s per iteration.

## 6. Evaluation

`oceanking.tools.evaluate_jump`: one environment, native MuJoCo on CPU, no observation noise, no
startup randomisation; beta 0.8, 0.9, 0.98 × targets straight up, full strength forward, sideways
(left), backward, half strength at 45°. Reported: landing point and error, rise, airtime, falls;
official (expanded) policy against candidates.
