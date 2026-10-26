# ocean.jump, run 5, iteration 11000 (selected policy: the straight-up jump)

Fine-tuned from the official `jumper.ref_free_jump` checkpoint (iteration 5999, expanded by
`oceanking.tools.expand_checkpoint --task jump`) for 3000 iterations
(`logs/jumper/ocean.jump/2026-10-26_13-12-51`) and 2000 more from its last checkpoint
(`2026-10-26_17-12-54`), 4096 environments, beta 0.75-0.98, code at commit `80164cc`
(docs/JUMP.md). Selected with the user as a **straight-up** jump: it does not aim (docs/JUMP.md
section 4), so it is commanded with strength 0.

```bash
cd jumper
../.venv/bin/python scripts/play.py --task ocean.jump \
    --checkpoint ../oceanking/tasks/ocean/jump/out/run5/model_11000.pt \
    --buoyancy 0.98 --ground-friction 0.6 --jump 0 0
```

`oceanking.tools.evaluate_jump`, straight up, against the official policy (landing at touchdown):

| beta | rise (official) | landing error (official) | airtime (official) |
|---|---|---|---|
| 0.8 | 0.22 m (0.16) | 0.01 m (0.07) | 1.0 s (0.9) |
| 0.9 | 0.24 m (0.20) | 0.00 m (0.12) | 1.5 s (1.5) |
| 0.98 | 0.26 m (0.30) | 0.02 m (0.05) | 4.1 s (4.9) |

Upright throughout (minimum body-up component 0.98). Higher than the official jump at beta 0.8 and
0.9, slightly lower at 0.98, and it lands where it took off.
