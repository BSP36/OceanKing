"""Drag calibration: MuJoCo's per-geom drag set against each part's outline.

MuJoCo's ellipsoid fluid model computes drag for each geom on its own, as if it
were alone in the flow. Summed over Jumper, that overstates drag about threefold
-- but unevenly: the torso is two overlapping hulls and the crown a cluster of
small parts, both far above a bluff body's drag, while a leg on its own is close
to a cylinder's. One factor for the whole robot would cut the legs, whose swing
through the water is the main hydrodynamic load on the servos, to half of what
a cylinder feels.

So the robot is split into **groups** (torso, crown, legs, arms) and each group's
geoms get their own multiplier (`fluid_ellipsoid`, which scales a geom's whole
fluid force and torque on both simulators), chosen so that the group alone, in a
head-on flow, has drag coefficient ``CD_TARGET[group]`` on its reference area.
The torso and the crown are each one bluff body: the reference is their
silhouette. A limb is a separate cylinder a few diameters from the next, which
shields it little: the reference is the **sum of each limb's own silhouette**.
Measured first against the legs' joint silhouette, where front legs hide rear
ones, the legs read Cd 4.0 and would have been cut to a quarter.
Shape-dependence (side-on, heave, rotation) stays MuJoCo's. The water's density
is left at its real value.

`measure()` takes the measurement; `python -m oceanking.tools.drag_calibration`
stores it in `oceanking/data/jumper_drag.json`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "jumper_drag.json"

#: Head-on drag coefficient each group should have alone, on its own frontal
#: silhouette, at Re ~ 1e4-1e5. Torso: a rounded box (a cube is ~1.05). Crown: a
#: short upright cylinder with points. Limbs: cylinders across the flow, 1.1-1.2.
CD_TARGET = {"torso": 1.0, "crown": 1.0, "legs": 1.1, "arms": 1.1}
#: Water speed the calibration is measured at, m/s; drag is quadratic here, so
#: the factors hardly depend on it (the viscous term is ~1e-4 of the total).
SPEED = 0.3
#: Rasterisation step for silhouettes, m.
PIXEL = 0.001

_TORSO = {"base_link", "upper_shell_link", "display_module_link", "tof_sensor_link",
          "camera_link"}


def unit_of(link: str) -> str:
    """The rigid unit a link's outline belongs to: one limb, or the group itself."""
    return link[:2] if group_of(link) in ("legs", "arms") else str(group_of(link))


def group_of(link: str) -> str | None:
    """The drag group of a body (its name without the entity prefix)."""
    if link in _TORSO:
        return "torso"
    if link == "oceanking_crown":
        return "crown"
    if link[:2] in ("LM", "RM", "LR", "RR"):
        return "legs"
    if link[:2] in ("LF", "RF"):
        return "arms"
    return None


def _fingerprint() -> dict:
    from oceanking import crown
    from oceanking.buoyancy import link_table

    return {"model_sha256": link_table()["model_sha256"],
            "crown_outer_volume_m3": round(crown.outer_volume(), 9),
            "cd_target": CD_TARGET}


@lru_cache(maxsize=1)
def _table() -> dict:
    table = json.loads(DATA.read_text(encoding="utf-8"))
    if table["fingerprint"] != _fingerprint():
        raise RuntimeError(
            f"{DATA.name} was measured on another model, crown or target. Re-run "
            "`.venv/bin/python -m oceanking.tools.drag_calibration`."
        )
    return table


def group_factors() -> dict[str, float]:
    return {g: float(e["factor"]) for g, e in _table()["groups"].items()}


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
    """Per-group raw drag of the standing kitted robot and the resulting factors."""
    import mujoco

    from oceanking.hydro import HydroCfg, apply_to_spec
    from oceanking.tools.link_volumes import home_pose

    rho = 1000.0
    spec = mujoco.MjSpec.from_file(str(xml))
    # Uniform multiplier 1: MuJoCo's raw per-geom model on every colliding geom.
    apply_to_spec(spec, HydroCfg(mode="full", buoyancy_ratio=0.8, drag_calibration=1.0,
                                 added_mass_coef=0.0))
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[2] = 0.107
    for joint, q in home_pose().items():
        d.qpos[m.jnt_qposadr[m.joint(joint).id]] = q
    mujoco.mj_forward(m, d)  # geom poses for the silhouettes
    fluid = m.geom_fluid.reshape(m.ngeom, -1)
    raw_flag = fluid[:, 0].copy()
    groups: dict[str, list[int]] = {}
    units: dict[str, dict[str, list[int]]] = {}
    for g in range(m.ngeom):
        link = m.body(m.geom_bodyid[g]).name.rsplit("/", 1)[-1]
        name = group_of(link)
        if name is not None and raw_flag[g] > 0:
            groups.setdefault(name, []).append(g)
            units.setdefault(name, {}).setdefault(unit_of(link), []).append(g)

    def drag(axis: int, scale: np.ndarray) -> float:
        current = np.zeros(3)
        current[axis] = -SPEED
        m.opt.wind[:] = current
        fluid[:, 0] = scale
        mujoco.mj_forward(m, d)
        return float(-d.qfrc_fluid[axis])  # free root: total force on the robot

    q = 0.5 * rho * SPEED**2
    out: dict[str, Any] = {"groups": {}}
    factor = np.zeros(m.ngeom)
    for name, geoms in sorted(groups.items()):
        # The other groups keep the flag at a negligible multiplier rather than 0:
        # a body with no flagged geom falls back to MuJoCo's inertia-box model and
        # would add its own drag (found when the groups summed to twice the whole).
        only = np.where(raw_flag > 0, 1e-12, 0.0)
        only[geoms] = raw_flag[geoms]
        area = sum(silhouette_area(m, d, u, 0) for u in units[name].values())
        raw_cd = drag(0, only) / q / area
        k = CD_TARGET[name] / raw_cd
        factor[geoms] = k * raw_flag[geoms]
        out["groups"][name] = {"geoms": len(geoms), "units": sorted(units[name]),
                               "raw_force_head_on_N": raw_cd * q * area,
                               "reference_area_m2": area, "raw_Cd": raw_cd,
                               "target_Cd": CD_TARGET[name], "factor": k}
    out["raw_force_head_on_N"] = drag(0, raw_flag)
    solid = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    for axis, name in ((0, "head_on_x"), (1, "side_on_y"), (2, "heave_z")):
        area = silhouette_area(m, d, solid, axis)
        out[name] = {"silhouette_m2": area,
                     "raw_Cd": drag(axis, raw_flag) / q / area,
                     "calibrated_Cd": drag(axis, factor) / q / area}
    return out
