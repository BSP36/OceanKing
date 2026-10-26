# ocean.walk, run 4, iteration 75600 (selected policy)

Fine-tuned from the official `jumper.posture` checkpoint (iteration 74800, expanded by
`oceanking.tools.expand_checkpoint`) for 800 iterations of `ocean.walk` on `seabed_rock`, 4096
environments, code at commit `cc2ed7f` (docs/TRAINING.md section 6.1, "Run 3 result and run 4").

```bash
cd jumper
../.venv/bin/python scripts/play.py --task ocean.walk --scene seabed_rock \
    --checkpoint ../oceanking/tasks/ocean/walk/out/run4/model_75600.pt \
    --buoyancy 0.8 --ground-friction 0.5 --current -0.1 0
```

Fixed evaluation (docs/TRAINING.md section 9), against the official policy: **passes all five
criteria**.

| Check | Value | Threshold |
|---|---|---|
| A, small commands | 1.18x | >= 0.95x |
| B, geometric mean speed ratio | 1.30x | >= 1.0x |
| C, into head currents | 1.53x | >= 1.05x |
| D, standing drift excess | +0.004 m/s | <= +0.005 m/s |
| E, minimum uprightness | 0.998 | > 0.5 |

Why this checkpoint, not the last (76299): every run-4 checkpoint from 75900 on also passes, with
higher scores, but they **overshoot** small commands (0.15 m/s forward walks 0.23 m/s at 76299), which
the frozen criteria (all "at least the official") do not catch, and they lose sideways speed in a
cross current. 75600 has the lowest mean relative error against reachable commands (0.15 and 0.3 m/s,
forward, sideways and backward, still water on rock, easy rock and sand): 0.19, against 0.31 for the
official policy and 0.30 for the last checkpoint. Full table: [evaluation.md](evaluation.md).
