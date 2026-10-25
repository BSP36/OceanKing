"""Print the physical quantities `docs/PHYSICS.md` quotes, from the compiled model.

    .venv/bin/python -m oceanking.tools.physics_report [--beta 0.8] [--crown-air 0.5]
"""

from __future__ import annotations

import argparse
import math

import mujoco
import numpy as np

from oceanking import buoyancy as B
from oceanking import crown, drag
from oceanking.hydro import HydroCfg, apply_to_spec
from oceanking.tools.link_volumes import XML, home_pose

G = 9.81
RHO = 1000.0
FOOT_FRICTION = 1.2  # upstream's nominal foot friction (tasks/jumper/common/constants.py)


def _model(cfg: HydroCfg) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = mujoco.MjSpec.from_file(str(XML))
    apply_to_spec(spec, cfg)
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[2] = 0.107
    for joint, q in home_pose().items():
        d.qpos[m.jnt_qposadr[m.joint(joint).id]] = q
    mujoco.mj_forward(m, d)
    return m, d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--crown-air", type=float, default=0.5)
    args = ap.parse_args()

    cfg = HydroCfg(mode="buoyancy", buoyancy_ratio=args.beta, crown_air=args.crown_air)
    ballast = cfg.ballast()
    m, d = _model(cfg)
    M, V = ballast.total_mass, ballast.total_volume
    print(f"== beta {args.beta}, crown air {args.crown_air}")
    print(f"gravity                {m.opt.gravity}")
    print(f"physical mass          {M:.3f} kg (robot {ballast.robot_mass:.3f}, keel "
          f"{ballast.keel_mass:.3f}, crown {ballast.crown_mass:.3f})")
    print(f"displaced volume       {V * 1e3:.3f} L -> buoyancy {RHO * V * G:.2f} N")
    print(f"weight                 {M * G:.2f} N; net (model) {B.net_weight(m, 'base_link'):.2f} N "
          f"= (1 - beta) M g {(1 - args.beta) * M * G:.2f} N")

    # Centres of mass and buoyancy (physical bodies only; buoyancy points carry lift).
    lift = np.zeros(m.nbody)
    for b in range(m.nbody):
        if m.body(b).name.endswith("_cob"):
            lift[b] = (m.body_gravcomp[b] - 1.0) * m.body_mass[b]
        elif m.body(b).name == "oceanking_crown":
            lift[b] = m.body_gravcomp[b] * m.body_mass[b]
    phys = np.array([not m.body(b).name.endswith("_cob") for b in range(m.nbody)])
    com = (m.body_mass[phys, None] * d.xipos[phys]).sum(0) / m.body_mass[phys].sum()
    cob = (lift[:, None] * d.xipos).sum(0) / lift.sum()
    bg = cob[2] - com[2]
    print(f"centre of mass z       {com[2] * 1e3:.1f} mm; centre of buoyancy z {cob[2] * 1e3:.1f} mm"
          f"; BG {bg * 1e3:.1f} mm (positive: stable)")
    print(f"  horizontal B-G       {((cob - com)[:2] * 1e3).round(1)} mm")
    print(f"  righting moment      {RHO * V * G * bg:.3f} N m per rad of roll or pitch (small angles)")

    # Drag.
    table = drag._table()
    for axis in ("head_on_x", "side_on_y", "heave_z"):
        e = table[axis]
        print(f"drag {axis:10s}      Cd {e['calibrated_Cd']:.2f} on {e['silhouette_m2'] * 1e4:.0f} cm2"
              f" -> {0.5 * RHO * e['calibrated_Cd'] * e['silhouette_m2'] * 0.3**2:.2f} N at 0.3 m/s")
    heave = table["heave_z"]
    vt = math.sqrt(2 * (1 - args.beta) * M * G / (RHO * heave["calibrated_Cd"] * heave["silhouette_m2"]))
    print(f"terminal sink speed    {vt:.2f} m/s (net weight against heave drag)")

    # Added mass.
    added = 0.5 * RHO * V
    print(f"added mass (C_a 0.5)   {added:.2f} kg = {added / M:.0%} of M; sink acceleration "
          f"{(1 - args.beta) * M * G / (M + added):.2f} m/s^2 (without: {(1 - args.beta) * G:.2f})")

    # Traction against drag.
    cap = FOOT_FRICTION * (1 - args.beta) * M * G
    head = table["head_on_x"]
    v_lim = math.sqrt(2 * cap / (RHO * head["calibrated_Cd"] * head["silhouette_m2"]))
    print(f"traction (mu {FOOT_FRICTION})       {cap:.2f} N -> head-on relative water speed limit "
          f"{v_lim:.2f} m/s")
    print(f"crown                  outer {crown.outer_volume() * 1e3:.3f} L, tank "
          f"{crown.tank_volume() * 1e3:.3f} L, dry {crown.dry_mass():.3f} kg")


if __name__ == "__main__":
    main()
