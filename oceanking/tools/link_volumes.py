"""Measure Jumper's displaced volume per link and write `oceanking/data/jumper_link_volumes.json`.

The volume of a link is the convex hull of its collision mesh (the `group=1`
`*_meshcol` geom of `jumper.xml`): every link is treated as a **sealed** housing
whose outer shell displaces water, which is what an air-filled robot does.

Neighbouring hulls overlap (the upper shell sits inside the base's hull, joints
share material), so summing hulls counts some water twice. Each link is therefore
credited only with the part of its hull that no earlier link (in body order:
the torso first) already covers, estimated by sampling points uniformly inside
the hull. The credited volumes sum to the volume of the union. Measured at the
HOME pose; leg motion changes the small joint overlaps, not the total.

Run from the repository root; `oceanking.hydro` refuses a table whose model hash
does not match the asset it is applied to.

    .venv/bin/python -m oceanking.tools.link_volumes
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

REPO = Path(__file__).resolve().parents[2]
XML = REPO / "jumper" / "assets" / "jumper" / "jumper.xml"
OUT = REPO / "oceanking" / "data" / "jumper_link_volumes.json"
COLLISION_GROUP = 1


def model_hash(xml: Path) -> str:
    return hashlib.sha256(xml.read_bytes()).hexdigest()


def home_pose() -> dict[str, float]:
    import importlib

    return importlib.import_module("tasks.jumper.common.constants").HOME


def measure(xml: Path, samples_per_link: int, seed: int) -> dict:
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    for name, q in home_pose().items():
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = q
    mujoco.mj_forward(m, d)

    hulls: list[tuple[int, np.ndarray]] = []
    for g in range(m.ngeom):
        if m.geom_group[g] != COLLISION_GROUP or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        world = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        hulls.append((int(m.geom_bodyid[g]), world))
    hulls.sort(key=lambda h: h[0])
    if len({b for b, _ in hulls}) != len(hulls):
        raise SystemExit("expected exactly one collision mesh per body")

    rng = np.random.default_rng(seed)
    tris = [Delaunay(w) for _, w in hulls]
    links = {}
    for i, (body, world) in enumerate(hulls):
        hull_volume = ConvexHull(world).volume
        lo, hi = world.min(0), world.max(0)
        pts = np.empty((0, 3))
        while len(pts) < samples_per_link:
            p = lo + rng.random((samples_per_link, 3)) * (hi - lo)
            pts = np.vstack([pts, p[tris[i].find_simplex(p) >= 0]])
        pts = pts[:samples_per_link]
        covered = np.zeros(len(pts), bool)
        for j in range(i):
            covered |= tris[j].find_simplex(pts) >= 0
        own = 1.0 - covered.mean()
        links[m.body(body).name] = {
            "mass_kg": float(m.body_mass[body]),
            "hull_volume_m3": float(hull_volume),
            "volume_m3": float(hull_volume * own),
        }
    # Torso bounds in the base frame (the free joint sits at the origin, unrotated,
    # so world and base frames coincide here): where the kit is mounted.
    torso = np.vstack([w for b, w in hulls
                       if m.body(b).name in ("base_link", "upper_shell_link")])
    total_mass = float(m.body_mass.sum())
    total_volume = sum(v["volume_m3"] for v in links.values())
    return {
        "model": str(xml.relative_to(REPO)),
        "model_sha256": model_hash(xml),
        "method": "convex hull of each link's group-1 collision mesh at HOME; overlap "
        "credited to the earlier body; Monte Carlo inside each hull",
        "samples_per_link": samples_per_link,
        "seed": seed,
        "total_mass_kg": total_mass,
        "total_volume_m3": total_volume,
        "sealed_buoyancy_ratio_fresh_water": 1000.0 * total_volume / total_mass,
        "torso_bounds_m": {"min": torso.min(0).tolist(), "max": torso.max(0).tolist()},
        "links": links,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    table = measure(XML, args.samples, args.seed)
    args.out.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(f"total mass {table['total_mass_kg']:.4f} kg, sealed volume "
          f"{table['total_volume_m3'] * 1e3:.4f} L, sealed beta "
          f"{table['sealed_buoyancy_ratio_fresh_water']:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
