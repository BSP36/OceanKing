"""Turn the official `jumper.posture` checkpoint into `ocean.walk`'s starting point.

`ocean.walk`'s actor sees upstream's 415 observation values plus the water's 4
(beta, the current in the heading frame), and its critic upstream's 602 plus 5
(those and the ground's friction), appended last (docs/TRAINING.md section 3).
This widens the first layer of both networks with **zero columns**, so the
expanded policy computes exactly what the official one does whatever the water
reads, and extends everything sized by the input with it:

- the observation normalisers, with the new columns' mean and variance set from
  the training ranges rather than to (0, 1): the normaliser's count is in the
  millions, so whatever it starts at for a new column is where it stays;
- Adam's moments for the first layer (zeros for the new columns), kept so the
  file is a complete checkpoint; it is marked `oceanking_fresh_optimizer` and
  ocean.walk's runner does not restore them (a fresh Adam);
- `iter` and `infos` (curricula and the environment's step counter), so a
  resumed run starts at the official command curriculum's top rung.

    .venv/bin/python -m oceanking.tools.expand_checkpoint OUT_DIR

then `scripts/train.py --task ocean.walk --checkpoint OUT_DIR/model_74800.pt`; with
`--task jump`, ocean.jump's from the official `jumper.ref_free_jump` checkpoint
(docs/JUMP.md section 5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
OFFICIAL = REPO / "jumper/tasks/jumper/posture/out/example/model_74800.pt"
#: ocean.jump's starting point (docs/JUMP.md section 5).
OFFICIAL_JUMP = REPO / "jumper/tasks/jumper/ref_free_jump/out/example/model_5999.pt"


def jump_columns() -> dict[str, list[tuple[str, float, float]]]:
    """(name, mean, variance) of each column ocean.jump appends (docs/JUMP.md
    section 3), from its ranges: the target is s (uniform, 0 in `UPRIGHT_SHARE`)
    times a uniform direction, so each component has mean 0 and variance
    E[s^2] / 2; the offset from the target starts as minus the target, in metres."""
    import tasks.ocean.jump.env_cfg as jump

    lo, hi = jump.BETA_RANGE
    beta = ("buoyancy_ratio", (lo + hi) / 2, (hi - lo) ** 2 / 12)
    t = (1 - jump.UPRIGHT_SHARE) / 3 / 2  # E[s^2] = 1/3 on the aimed share
    target = [("target_forward", 0.0, t), ("target_left", 0.0, t), ("target_up", 0.0, 1e-4)]
    m = t * jump.REACH**2
    offset = [("offset_forward", 0.0, m), ("offset_left", 0.0, m), ("offset_up", 0.0, 1e-4)]
    flo, fhi = jump.FRICTION_RANGE
    friction = ("ground_friction", (flo + fhi) / 2, (fhi - flo) ** 2 / 12)
    return {"actor": [*target, beta], "critic": [*target, beta, friction, *offset]}


def new_columns() -> dict[str, list[tuple[str, float, float]]]:
    """(name, mean, variance) of each appended column, from ocean.walk's ranges."""
    import tasks.ocean.walk.env_cfg as walk

    lo, hi = walk.BETA_RANGE
    beta = ("buoyancy_ratio", (lo + hi) / 2, (hi - lo) ** 2 / 12)
    # Speed uniform on [0, c], direction uniform: each horizontal component has
    # mean 0 and variance E[s^2] / 2 = c^2 / 6.
    c = walk.CURRENT_MAX
    horizontal = c**2 / 6
    noise = walk.CURRENT_NOISE**2 / 3  # uniform on [-n, n]
    flo, fhi = walk.FRICTION_RANGE
    # The vertical component is always 0 and carries no noise (noise goes on the
    # horizontal ones only); a small variance keeps the normaliser finite.
    actor = [beta, ("current_forward", 0.0, horizontal + noise),
             ("current_left", 0.0, horizontal + noise), ("current_up", 0.0, 1e-4)]
    critic = [beta, ("current_forward", 0.0, horizontal), ("current_left", 0.0, horizontal),
              # Never moves in training; a small variance keeps the normaliser finite.
              ("current_up", 0.0, 1e-4),
              ("ground_friction", (flo + fhi) / 2, (fhi - flo) ** 2 / 12)]
    return {"actor": actor, "critic": critic}


def _widen(state: dict, cols: list[tuple[str, float, float]]) -> tuple[int, int]:
    w = state["mlp.0.weight"]
    old = w.shape[1]
    state["mlp.0.weight"] = torch.cat([w, w.new_zeros(w.shape[0], len(cols))], dim=1)
    mean = torch.tensor([[m for _, m, _ in cols]], dtype=w.dtype)
    var = torch.tensor([[v for _, _, v in cols]], dtype=w.dtype)
    state["obs_normalizer._mean"] = torch.cat([state["obs_normalizer._mean"], mean], dim=1)
    state["obs_normalizer._var"] = torch.cat([state["obs_normalizer._var"], var], dim=1)
    state["obs_normalizer._std"] = torch.cat([state["obs_normalizer._std"], var.sqrt()], dim=1)
    return old, old + len(cols)


def expand(src: Path, dst: Path, cols: dict | None = None) -> dict:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    cols = cols or new_columns()
    widths = {"actor": _widen(ck["actor_state_dict"], cols["actor"]),
              "critic": _widen(ck["critic_state_dict"], cols["critic"])}
    for state in ck["optimizer_state_dict"]["state"].values():
        for key in ("exp_avg", "exp_avg_sq"):
            t = state[key]
            for old, new in widths.values():
                if t.dim() == 2 and t.shape[1] == old and t.shape[0] == 512:
                    state[key] = torch.cat([t, t.new_zeros(512, new - old)], dim=1)
                    break
    # The Adam moments above are widened to fit, but they describe the land task and
    # gave the new columns several times the normal step at first (review of run
    # 2): ocean.walk's runner starts a fresh optimiser for a checkpoint marked so.
    ck["oceanking_fresh_optimizer"] = True
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / src.name
    torch.save(ck, out)
    record = {
        "source": str(src.relative_to(REPO)) if src.is_relative_to(REPO) else str(src),
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "iteration": ck["iter"],
        "widths": {k: {"from": a, "to": b} for k, (a, b) in widths.items()},
        "new_columns": {k: [{"name": n, "mean": m, "variance": v} for n, m, v in c]
                        for k, c in cols.items()},
    }
    (dst / "expanded_from.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path, help="directory to write the expanded checkpoint into")
    ap.add_argument("--task", choices=("walk", "jump"), default="walk")
    ap.add_argument("--source", type=Path, default=None)
    args = ap.parse_args()
    if args.source is None:
        args.source = OFFICIAL_JUMP if args.task == "jump" else OFFICIAL
    record = expand(args.source, args.out, jump_columns() if args.task == "jump" else None)
    print(json.dumps(record["widths"]), "->", args.out / args.source.name)


if __name__ == "__main__":
    main()
