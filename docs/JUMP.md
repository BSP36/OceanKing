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

Upstream's terms and weights, with these changes (as of run 4):

| Term | Upstream | ocean.jump |
|---|---|---|
| `landing_target` (new, +6) | | `exp(-|landing - target|^2 / 0.08^2)`, paid every step after the episode's own jump has landed, nothing after a second jump |
| `landing_near` (new, +4) | | the same with a 0.25 m kernel: still pays a full-strength jump that lands 0.3 m off (0.24) |
| `apex_height` (+4) | rise of the lowest point / 0.1 m, after landing | the same **times the wide landing accuracy**: height counts on a jump that lands roughly where aimed |
| `flight_progress` (new, +10) | | in the episode's own flight, aimed episodes only: the speed towards the target (clipped ±1 m/s), i.e. the rate the distance to it shrinks; summed over a flight, the distance gained (bounded by the target's; passing it turns the sign). Nothing on the ground |
| `flight_height` (+2) | rise during flight | **0**: it pays every airborne step, and at beta 0.98 a flight lasts seconds, which would pay more than any landing |
| `horizontal_drift` (−1) | distance from the start | `anchored_drift`: beyond 0.05 m from the take-off point before the jump, nothing in flight, beyond 0.05 m from the landing point after it (no shuffling towards the target, no sliding after landing) |

**Run 1 (stopped at iteration 6650 of 9000)** had only the narrow kernel, scaled the apex by it,
and charged all travel before take-off. At its iteration 6500 the policy landed within 0.06 m of
its take-off point whatever the target (mean error 0.25 m over aimed targets, the official 0.27 m)
and jumped lower (rise 0.12 m at beta 0.8, the official 0.16 m): a missed full-strength target paid
`exp(-14)`, nothing to climb, while not aiming kept the drift penalty and the apex scaling quiet.

**Run 2 (stopped at 6500)**, with the wide kernel and the deadband, still ignored the target: at
6500 it landed within ±0.1 m of its take-off point in every direction, the landing point set by
beta alone. The landing terms are paid seconds after the push and the landing scatter is small, so
the target barely moved the return. Run 3 adds `flight_progress`, a dense signal during the flight
itself.

**Run 3 (stopped at 6450) and the bug behind runs 1–3.** Every episode of run 3 ran the full 8 s
and the jumps got lower. A probe of the command's state (`scratch/jump/probe/probe.py`) found two
faults in how upstream's jump detection reads the water, not in the rewards:

- **contacts against the land threshold**: a foot is "in contact" above 1 N. A standing robot
  puts 0.8 N on a foot at beta 0.9 and 0.15 N at 0.98, so above about beta 0.85 (more than half the
  training range) the robot read as airborne while standing: the jump "took off" at the spawn and
  never landed;
- **the two-stage push**: under water the official push starts with a ~1 cm hop, and the middle legs
  touch down again 0.3 s in while the body rises. Upstream's logic (1 cm minimum rise) latched that
  touch as the landing, at the take-off point, and the real flight as a second jump whose height it
  clawed back.

So the landing terms were scored at the take-off point, and in most environments never. Fixed
(`OceanJumpCommand._update_command`): the contact threshold is upstream's 1 N times the
environment's net weight over the land weight (floor 0.02 N), contacts while the body still rises
faster than 0.05 m/s are part of the push, and a jump must lift the lowest point 0.04 m
(`MIN_RISE`). Probed after the fix: beta 0.8 lands at 1.35 s, beta 0.9 at 1.85 s and the episode
then ends at HOME, beta 0.98 flies its whole 4 s arc. The body-on-ground contact
(`illegal_contact`) is judged against the net weight too.

## 5. Training

Fine-tuned from the official checkpoint, iteration 5999, widened with zero columns
(`expand_checkpoint --task jump`): it acts exactly as the official policy, whatever the new columns
read. At that iteration upstream's training aid (the recording fed forward, and the prior on the
joints) is already gone, and the environment's step counter is restored with it.

**Run 4 (stopped at 7050)**, with detection fixed, began to aim at its iteration 6500 (forward
targets landed 0.10–0.17 m forward, straight-up ones near the take-off point) but then lowered its
jumps (rise 0.15 → 0.09 m at 7000), the erosion every run showed. The cause is the discount:
upstream's horizon is ~2 s (0.99 and 0.95 at 50 Hz, fourth roots at 200 Hz), set for a land jump
whose push and landing are 0.5 s apart. Under water the flight lasts 1–5 s, so the landing's reward
reached the push discounted to ~0.13 while the push's costs were immediate. From run 5 the
horizon is ~6 s (twelfth roots), and the drift after landing has the same 0.05 m deadband (run 4
paid −1.1 per episode for the light robot settling, as much as the aiming terms paid).

PPO as ocean.walk's fine-tuning (docs/TRAINING.md section 6.4): fresh Adam at lr 1e-4, the action
std fixed at the official values, no entropy bonus; `desired_kl` stays at upstream's 0.01, since an
aimed jump is a new skill rather than a correction. 4096 environments: 1.2 s per iteration.

**Run 5 result (2026-10-26).** 3000 iterations, then 3000 more from its last checkpoint
(`2026-10-26_13-12-51`, `2026-10-26_17-12-54`). The jump itself is stable and as high as the
official one (rise 0.23–0.25 m mean over beta 0.8–0.98; the official 0.22 m), lands upright, and a
straight-up command lands within 0.02 m of its take-off point. **It does not aim**: every target
lands within 0.06 m of the take-off point (mean error over aimed targets 0.25 m; the official
0.28 m). `flight_progress` stayed at zero: the flights have no horizontal speed to reward, and
per-step exploration noise (white, std 0.24 rad) averages out over a 0.2 s push, so a coordinated
lean that would send a jump sideways is never tried.

**Feasibility of an aimed jump (open-loop probe, 2026-10-26, `scratch/jump/lean/`).** The official
policy (expanded checkpoint) with a fixed joint-target offset added during its first 0.4 s, beta
0.9, straight up; displacement over the longest flight:

| Offset | flight dx | rise |
|---|---|---|
| none | −0.113 m | 0.198 m |
| every J0 (leg sweep) ±0.05–0.1 rad | −0.09 to −0.12 m | 0.20 m |
| every J0 ±0.2–0.4 rad | — | 0.02–0.06 m (no jump) |
| front J1 +0.2 rad (pitch) | −0.018 m | 0.181 m |
| rear J1 +0.2 rad | −0.075 m | 0.194 m |
| front J1 +0.4 rad, or front and rear ±0.2 | — | 0.01–0.03 m (no jump) |

Pitching the body moves the landing by about 0.1 m with the jump intact; anything larger breaks
the jump of a policy that was never trained for it. With the take-off speeds the policy reaches
under water (0.5–0.8 m/s vertical) and the drag of section 2, an aimed jump is plausibly a
0.1–0.15 m affair, not 0.3 m.

## 6. Evaluation

`oceanking.tools.evaluate_jump`: one environment, native MuJoCo on CPU, no observation noise, no
startup randomisation; beta 0.8, 0.9, 0.98 × targets straight up, full strength forward, sideways
(left), backward, half strength at 45°. Reported: landing point (the touchdown at the end of the longest airborne stretch,
as training latches it, not the episode's last position: the official policy shuffles 0.24 m
after its jump at beta 0.8, which an earlier version of this tool counted as the jump) and error,
movement after landing, rise, airtime, falls;
official (expanded) policy against candidates.
