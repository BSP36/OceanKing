"""The fixed evaluation of an ocean.walk policy against the official one.

Frozen before run 3 (docs/TRAINING.md section 9): the same battery, the same
acceptance criteria, for every candidate. Every run is a native-MuJoCo replay of
one environment from the nominal start (`--nominal --no-obs-noise`), in the
ocean.walk task with the water pinned, so the official policy (through its
expanded checkpoint, which acts identically) and a candidate see the same world.

    .venv/bin/python -m oceanking.tools.evaluate OUT_DIR CANDIDATE.pt [CANDIDATE2.pt ...]

writes OUT_DIR/<label>/... trajectories and OUT_DIR/report.json / report.md.

Battery:
- walking: after 1 s standing, 4 s each of forward 0.15, 0.3, 0.8, sideways
  0.15, 0.3, 0.8 and backward 0.3 m/s (measured over each segment's last 2.5 s,
  body frame, along the command), in seven water conditions;
- standing: 9 s commanded to stand, from rest, in a current (drift over 3-9 s).

Acceptance (all must hold, each against the official policy in the same runs):
- A, small commands: forward and sideways 0.15 in the reference water, >= 0.95x;
- B, overall: geometric mean of the speed ratio over every walking segment >= 1.0;
- C, against the current: mean forward speed in the two head currents >= 1.05x;
- D, holding station: every standing drift <= the official's + 0.005 m/s;
- E, no falls (body z-axis . world z stays above 0.5).
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
OFFICIAL = REPO / "scratch" / "official_expanded" / "model_74800.pt"

SEGMENTS = [("fwd0.15", (0.15, 0, 0)), ("fwd0.3", (0.3, 0, 0)), ("fwd0.8", (0.8, 0, 0)),
            ("side0.15", (0, 0.15, 0)), ("side0.3", (0, 0.3, 0)), ("side0.8", (0, 0.8, 0)),
            ("back0.3", (-0.3, 0, 0))]
SEG_S = 4.0
LEAD_S = 1.0
STAND_S = 9.0

#: name -> (scene, beta, friction, current (vx, vy))
WALK = {
    "ref_still": ("seabed_rock", 0.8, 0.5, (0.0, 0.0)),
    "ref_head0.1": ("seabed_rock", 0.8, 0.5, (-0.1, 0.0)),
    "ref_head0.2": ("seabed_rock", 0.8, 0.5, (-0.2, 0.0)),
    "ref_cross0.2": ("seabed_rock", 0.8, 0.5, (0.0, -0.2)),
    "easy_still": ("seabed_rock", 0.75, 0.8, (0.0, 0.0)),
    "hard_still": ("seabed_rock", 0.9, 0.4, (0.0, 0.0)),
    "sand_still": ("seabed_sand", 0.8, 0.6, (0.0, 0.0)),
}
STAND = {
    "stand_ref_head0.2": ("seabed_rock", 0.8, 0.5, (-0.2, 0.0)),
    "stand_ref_cross0.2": ("seabed_rock", 0.8, 0.5, (0.0, -0.2)),
    "stand_hard_head0.15": ("seabed_rock", 0.9, 0.4, (-0.15, 0.0)),
}


def _scripts(out: Path) -> tuple[Path, Path]:
    walk = out / "cmd_walk.py"
    walk.write_text(
        "_N = (0.0, 0.0, 0.0, float('nan'))\n"
        f"SEG = {SEGMENTS!r}\n"
        f"def script(t):\n"
        f"    i = int((t - {LEAD_S}) // {SEG_S}) if t >= {LEAD_S} else -1\n"
        f"    v = SEG[i][1] if 0 <= i < len(SEG) else (0.0, 0.0, 0.0)\n"
        f"    return {{'twist': tuple(float(x) for x in v), 'posture': _N}}\n")
    stand = out / "cmd_stand.py"
    stand.write_text("def script(t):\n    return {'twist': (0.0, 0.0, 0.0), "
                     "'posture': (0.0, 0.0, 0.0, float('nan'))}\n")
    return walk, stand


def _run(ckpt: Path, stem: Path, script: Path, cond, seconds: float) -> Path:
    scene, beta, mu, (vx, vy) = cond
    steps = int(round(seconds / 0.005))
    cmd = [str(PYTHON), "scripts/play.py", "--task", "ocean.walk", "--checkpoint", str(ckpt),
           "--backend", "native", "--device", "cpu", "--num_envs", "1", "--headless",
           "--no-obs-noise", "--scene", scene, "--nominal", "--steps", str(steps),
           "--command-script", str(script), "--trajectory-out", str(stem),
           "--buoyancy", str(beta), "--ground-friction", str(mu), "--current", str(vx), str(vy)]
    log = stem.parent / (stem.name + ".log")
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=JUMPER, stdout=f, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, "MUJOCO_GL": "egl", "CUDA_VISIBLE_DEVICES": "",
                            "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    return stem.parent / (stem.name + ".npz")


def _yaw(q):
    return np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def _upright(q):
    return 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)


def walk_speeds(npz: Path) -> tuple[dict, float]:
    z = np.load(npz)
    t, p, q = z["t"], z["root_pos"][0], z["root_quat_wxyz"][0]
    yaw = np.unwrap(_yaw(q))
    out = {}
    for i, (name, v) in enumerate(SEGMENTS):
        end = LEAD_S + (i + 1) * SEG_S
        a, b = np.searchsorted(t, end - 2.5), np.searchsorted(t, end - 0.05)
        d = (p[b, :2] - p[a, :2]) / (t[b] - t[a])
        c, s = math.cos(yaw[a]), math.sin(yaw[a])
        body = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        direction = np.array(v[:2], float) / np.linalg.norm(v[:2])
        out[name] = float(body @ direction)
    return out, float(_upright(q).min())


def stand_drift(npz: Path) -> tuple[float, float]:
    z = np.load(npz)
    t, p, q = z["t"], z["root_pos"][0], z["root_quat_wxyz"][0]
    a, b = np.searchsorted(t, 3.0), np.searchsorted(t, STAND_S - 0.05)
    return float(np.linalg.norm(p[b, :2] - p[a, :2]) / (t[b] - t[a])), float(_upright(q).min())


def evaluate_policy(ckpt: Path, label: str, out: Path, pool) -> dict:
    d = out / label
    d.mkdir(parents=True, exist_ok=True)
    walk, stand = _scripts(out)
    jobs = {}
    walk_s = LEAD_S + len(SEGMENTS) * SEG_S + 0.5
    for name, cond in WALK.items():
        jobs[name] = pool.submit(_run, ckpt, d / name, walk, cond, walk_s)
    for name, cond in STAND.items():
        jobs[name] = pool.submit(_run, ckpt, d / name, stand, cond, STAND_S)
    res = {"checkpoint": str(ckpt), "walk": {}, "stand": {}, "min_upright": 1.0}
    for name, job in jobs.items():
        npz = job.result()
        if name in WALK:
            speeds, up = walk_speeds(npz)
            res["walk"][name] = speeds
        else:
            drift, up = stand_drift(npz)
            res["stand"][name] = drift
        res["min_upright"] = min(res["min_upright"], up)
    return res


def judge(cand: dict, off: dict) -> dict:
    ref = cand["walk"]["ref_still"], off["walk"]["ref_still"]
    small = min(ref[0][k] / ref[1][k] for k in ("fwd0.15", "side0.15"))
    ratios = [cand["walk"][c][s] / off["walk"][c][s] for c in WALK for s, _ in SEGMENTS
              if off["walk"][c][s] > 0.02]
    geo = float(np.exp(np.mean(np.log(np.clip(ratios, 1e-3, None)))))
    heads = ("ref_head0.1", "ref_head0.2")
    fwd = [s for s, _ in SEGMENTS if s.startswith("fwd")]
    c_cur = float(np.mean([cand["walk"][h][s] for h in heads for s in fwd]))
    o_cur = float(np.mean([off["walk"][h][s] for h in heads for s in fwd]))
    against = c_cur / o_cur if o_cur > 0 else float("inf")
    drift_excess = max(cand["stand"][k] - off["stand"][k] for k in STAND)
    checks = {
        "A_small_commands": (small, small >= 0.95),
        "B_overall_geomean": (geo, geo >= 1.0),
        "C_against_current": (against, against >= 1.05),
        "D_drift_excess_m_s": (drift_excess, drift_excess <= 0.005),
        "E_min_upright": (cand["min_upright"], cand["min_upright"] > 0.5),
    }
    return {"checks": {k: {"value": v, "pass": bool(p)} for k, (v, p) in checks.items()},
            "pass": all(p for _, p in checks.values()),
            "score": geo * min(1.0, against) - max(0.0, drift_excess) * 10}


def _markdown(official: dict, cands: dict, verdicts: dict) -> str:
    lines = ["| condition | segment | official | " + " | ".join(cands) + " |",
             "|---|---|---|" + "---|" * len(cands)]
    for c in WALK:
        for s, _ in SEGMENTS:
            row = [f"{official['walk'][c][s]:.3f}"] + [f"{r['walk'][c][s]:.3f}" for r in cands.values()]
            lines.append(f"| {c} | {s} | " + " | ".join(row) + " |")
    for k in STAND:
        row = [f"{official['stand'][k]:.3f}"] + [f"{r['stand'][k]:.3f}" for r in cands.values()]
        lines.append(f"| {k} | drift | " + " | ".join(row) + " |")
    lines += ["", "| check | " + " | ".join(cands) + " |", "|---|" + "---|" * len(cands)]
    for check in next(iter(verdicts.values()))["checks"]:
        cells = [f"{v['checks'][check]['value']:.3f} {'PASS' if v['checks'][check]['pass'] else 'fail'}"
                 for v in verdicts.values()]
        lines.append(f"| {check} | " + " | ".join(cells) + " |")
    lines.append("| **overall** | " + " | ".join("**PASS**" if v["pass"] else "fail"
                                                for v in verdicts.values()) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("candidates", type=Path, nargs="+")
    ap.add_argument("--official", type=Path, default=OFFICIAL)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    # The replays run from jumper/: every path is made absolute first.
    args.out = args.out.resolve()
    args.candidates = [c.resolve() for c in args.candidates]
    args.official = args.official.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "report.json"
    cache = json.loads(report_path.read_text()) if report_path.exists() else {}
    with cf.ThreadPoolExecutor(args.workers) as pool:
        if "official" not in cache:
            cache["official"] = evaluate_policy(args.official, "official", args.out, pool)
        # One policy at a time, its runs in parallel (a policy submitted into the
        # pool it waits on would deadlock it).
        for ck in args.candidates:
            label = f"{ck.parent.name}_{ck.stem}"
            if label not in cache:
                cache[label] = evaluate_policy(ck, label, args.out, pool)
                report_path.write_text(json.dumps(cache, indent=2) + "\n")
    official = cache["official"]
    cands = {k: v for k, v in cache.items() if k not in ("official", "verdicts")}
    verdicts = {k: judge(v, official) for k, v in cands.items()}
    cache["verdicts"] = verdicts
    report_path.write_text(json.dumps(cache, indent=2) + "\n")
    md = _markdown(official, cands, verdicts)
    (args.out / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
