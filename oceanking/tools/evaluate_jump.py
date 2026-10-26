"""The fixed evaluation of an ocean.jump policy against the official jump (docs/JUMP.md section 6).

Every run is a native-MuJoCo replay of one environment without observation noise
or startup randomisation, in ocean.jump with beta, ground friction and the target
pinned; the official policy runs through its expanded checkpoint, which acts
identically to it. Only the first episode of each run is scored (play.py starts
another after `back_home`).

    .venv/bin/python -m oceanking.tools.evaluate_jump OUT_DIR CANDIDATE.pt [...]

writes OUT_DIR/<label>/... trajectories and OUT_DIR/report.json / report.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
JUMPER = REPO / "jumper"
PYTHON = REPO / ".venv" / "bin" / "python"
OFFICIAL = REPO / "scratch" / "official_jump_expanded" / "model_5999.pt"

REACH = 0.3
BETAS = (0.8, 0.9, 0.98)
FRICTION = 0.6
#: name -> (strength, direction in degrees, counter-clockwise from forward)
TARGETS = {"up": (0.0, 0.0), "fwd": (1.0, 0.0), "left": (1.0, 90.0),
           "back": (1.0, 180.0), "half45": (0.5, 45.0)}
SECONDS = 8.0
#: Above the start by this much, m, counts as airborne (for the airtime).
AIR_RISE = 0.03


def _run(ckpt: Path, stem: Path, beta: float, target: tuple[float, float]) -> Path:
    cmd = [str(PYTHON), "scripts/play.py", "--task", "ocean.jump", "--checkpoint", str(ckpt),
           "--backend", "native", "--device", "cpu", "--num_envs", "1", "--headless",
           "--no-obs-noise", "--nominal", "--steps", str(int(SECONDS / 0.005)),
           "--buoyancy", str(beta), "--ground-friction", str(FRICTION),
           "--jump", str(target[0]), str(target[1]), "--trajectory-out", str(stem)]
    with open(stem.parent / (stem.name + ".log"), "w") as f:
        subprocess.run(cmd, cwd=JUMPER, stdout=f, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, "MUJOCO_GL": "egl", "CUDA_VISIBLE_DEVICES": "",
                            "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    return stem.parent / (stem.name + ".npz")


def score(npz: Path, target: tuple[float, float]) -> dict:
    z = np.load(npz)
    t, p, q, ep = z["t"], z["root_pos"][0], z["root_quat_wxyz"][0], z["episode_step"][0]
    resets = np.where(np.diff(ep) < 0)[0]
    end = resets[0] + 1 if len(resets) else len(t)
    t, p, q = t[:end], p[:end], q[:end]
    z0 = p[0, 2]
    up = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    air = p[:, 2] > z0 + AIR_RISE
    s, deg = target
    goal = s * REACH * np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
    land = p[-1, :2]
    return {"landing": land.round(4).tolist(), "target": goal.round(4).tolist(),
            "error_m": float(np.linalg.norm(land - goal)),
            "rise_m": float(p[:, 2].max() - z0),
            "airtime_s": float(air.sum() * (t[1] - t[0])),
            "min_upright": float(up.min()), "episode_s": float(t[-1])}


def evaluate_policy(ckpt: Path, label: str, out: Path, pool) -> dict:
    d = out / label
    d.mkdir(parents=True, exist_ok=True)
    jobs = {(b, name): pool.submit(_run, ckpt, d / f"b{b}_{name}", b, tgt)
            for b in BETAS for name, tgt in TARGETS.items()}
    res = {"checkpoint": str(ckpt), "runs": {}}
    for (b, name), job in jobs.items():
        res["runs"][f"b{b}_{name}"] = score(job.result(), TARGETS[name])
    runs = res["runs"].values()
    aimed = [r for k, r in res["runs"].items() if not k.endswith("_up")]
    res["summary"] = {
        "mean_error_m": float(np.mean([r["error_m"] for r in runs])),
        "mean_error_aimed_m": float(np.mean([r["error_m"] for r in aimed])),
        "mean_rise_m": float(np.mean([r["rise_m"] for r in runs])),
        "mean_airtime_s": float(np.mean([r["airtime_s"] for r in runs])),
        "min_upright": float(min(r["min_upright"] for r in runs)),
    }
    return res


def _markdown(results: dict) -> str:
    labels = list(results)
    lines = ["| run | target (m) | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    for k, r0 in results[labels[0]]["runs"].items():
        cells = []
        for lab in labels:
            r = results[lab]["runs"][k]
            cells.append(f"({r['landing'][0]:+.2f}, {r['landing'][1]:+.2f}) err {r['error_m']:.2f}, "
                         f"rise {r['rise_m']:.2f}, air {r['airtime_s']:.1f}s")
        lines.append(f"| {k} | ({r0['target'][0]:+.2f}, {r0['target'][1]:+.2f}) | " + " | ".join(cells) + " |")
    lines += ["", "| summary | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for key in results[labels[0]]["summary"]:
        lines.append(f"| {key} | " + " | ".join(f"{results[lab]['summary'][key]:.3f}" for lab in labels) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("candidates", type=Path, nargs="*")
    ap.add_argument("--official", type=Path, default=OFFICIAL)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    # The replays run from jumper/: every path is made absolute first.
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    report = args.out / "report.json"
    cache = json.loads(report.read_text()) if report.exists() else {}
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for label, ck in [("official", args.official.resolve())] + [
                (f"{c.resolve().parent.name}_{c.stem}", c.resolve()) for c in args.candidates]:
            if label not in cache:
                cache[label] = evaluate_policy(ck, label, args.out, pool)
                report.write_text(json.dumps(cache, indent=2) + "\n")
    md = _markdown(cache)
    (args.out / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
