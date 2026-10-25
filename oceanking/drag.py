"""Drag calibration: what MuJoCo's per-geom model gives against the robot's outline.

MuJoCo's ellipsoid fluid model computes drag for each geom on its own, as if it
were alone in the flow. Jumper is 50-odd overlapping and closely spaced geoms, so
the sum counts the torso twice (base and upper shell hulls overlap) and every leg
as if nothing shielded it. Measured on the standing robot that sum is a drag
coefficient of about 3 on the frontal silhouette, where a rough bluff body of this
size at Re ~ 1e5 has about 1.

So the density the drag model sees is scaled by a single factor that brings the
head-on drag of the standing robot to ``CD_TARGET`` times its frontal silhouette
area. Shape-dependence (side-on, heave, rotation) stays MuJoCo's; only the level
is set. The measurement is `measure()`; its result is stored in
`oceanking/data/jumper_drag.json` by `python -m oceanking.tools.drag_calibration`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "jumper_drag.json"

#: Head-on drag coefficient on the frontal silhouette. Rough bluff bodies at
#: Re ~ 1e5 (open-frame ROVs, cubes, short cylinders in cross-flow) sit around
#: 0.8-1.3; 1.0 is the middle (docs/PHYSICS.md).
CD_TARGET = 1.0
#: Water speed the calibration is measured at, m/s; drag is quadratic here, so
#: the factor hardly depends on it (the viscous term is ~1e-4 of the total).
SPEED = 0.3
#: Rasterisation step for silhouettes, m.
PIXEL = 0.001


def _fingerprint() -> dict:
    from oceanking import crown
    from oceanking.buoyancy import link_table

    return {"model_sha256": link_table()["model_sha256"],
            "crown_outer_volume_m3": round(crown.outer_volume(), 9)}


@lru_cache(maxsize=1)
def _table() -> dict:
    table = json.loads(DATA.read_text(encoding="utf-8"))
    if table["fingerprint"] != _fingerprint():
        raise RuntimeError(
            f"{DATA.name} was measured on another model or crown. Re-run "
            "`.venv/bin/python -m oceanking.tools.drag_calibration`."
        )
    return table


def calibration_factor() -> float:
    return float(_table()["calibration_factor"])


def _geom_points(m: Any, d: Any, g: int) -> np.ndarray:
    import mujoco

    t = m.geom_type[g]
    s = m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        a = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        ring = np.c_[s[0] * np.cos(a), s[0] * np.sin(a)]
        v = np.vstack([np.c_[ring, np.full(len(a), z)] for z in (-s[1], s[1])])
    else:  # sphere, ellipsoid: points on the surface
        u = np.random.default_rng(0).normal(size=(2000, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        v = u * (s[0] if t == mujoco.mjtGeom.mjGEOM_SPHERE else s[:3])
    return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def silhouette_area(m: Any, d: Any, geoms: list[int], drop_axis: int) -> float:
    """Area of the union of the geoms' convex outlines seen along `drop_axis`."""
    from matplotlib.path import Path as Polygon
    from scipy.spatial import ConvexHull

    keep = [i for i in range(3) if i != drop_axis]
    polys = []
    for g in geoms:
        p = _geom_points(m, d, g)[:, keep]
        polys.append(Polygon(p[ConvexHull(p).vertices]))
    allp = np.vstack([p.vertices for p in polys])
    lo, hi = allp.min(0), allp.max(0)
    xs, ys = np.meshgrid(np.arange(lo[0], hi[0], PIXEL), np.arange(lo[1], hi[1], PIXEL))
    pts = np.c_[xs.ravel(), ys.ravel()]
    inside = np.zeros(len(pts), bool)
    for p in polys:
        inside |= p.contains_points(pts)
    return float(inside.sum() * PIXEL * PIXEL)


def measure(xml: Path) -> dict:
    """Raw (uncalibrated) drag of the standing kitted robot against its outline."""
    import mujoco

    from oceanking.hydro import HydroCfg, apply_to_spec
    from oceanking.tools.link_volumes import home_pose

    rho = 1000.0
    out: dict[str, Any] = {}
    for axis, name in ((0, "head_on_x"), (1, "side_on_y"), (2, "heave_z")):
        current = [0.0, 0.0, 0.0]
        current[axis] = -SPEED
        spec = mujoco.MjSpec.from_file(str(xml))
        apply_to_spec(spec, HydroCfg(mode="full", buoyancy_ratio=0.8, current=tuple(current),
                                     drag_calibration=1.0, added_mass_coef=0.0))
        m = spec.compile()
        d = mujoco.MjData(m)
        d.qpos[2] = 0.107
        for joint, q in home_pose().items():
            d.qpos[m.jnt_qposadr[m.joint(joint).id]] = q
        mujoco.mj_forward(m, d)
        force = -d.qfrc_fluid[axis]  # on the free root: the total force on the robot
        solid = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        area = silhouette_area(m, d, solid, axis)
        cda = force / (0.5 * rho * SPEED**2)
        out[name] = {"silhouette_m2": area, "raw_CdA_m2": cda, "raw_Cd": cda / area}
    out["calibration_factor"] = CD_TARGET / out["head_on_x"]["raw_Cd"]
    for name in ("head_on_x", "side_on_y", "heave_z"):
        out[name]["calibrated_Cd"] = out[name]["raw_Cd"] * out["calibration_factor"]
    return out
