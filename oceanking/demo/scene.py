"""The demo's seabed: sunken trash to collect and a place to bring it (docs/DEMO.md).

Trash is what sinks and stays put: a flooded can, a flooded glass bottle and a
waterlogged box, each a plain solid of one colour the onboard camera can pick
out, with an **effective** density -- the material and the water trapped inside,
averaged over the outer shape -- so that its weight under water is what the real
thing's is to first order. Each gets Archimedes' buoyancy (`oceanking.props`) and
the fluid model's drag, and the contact model of upstream's pick-place row
(`five_foot/objects.py`: friction, condim, solref, collision channel), so the claw
meets them as it meets those props.

The drop zone is a marker, not an obstacle: a flat magenta disk and a pole, both
visual only, fixed to the ground.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco


@dataclass(frozen=True)
class Trash:
    name: str
    kind: str  # "cylinder" (standing) or "box"
    size: tuple[float, ...]  # cylinder (radius, half-height); box half-extents
    density: float  # effective, kg/m^3
    rgba: tuple[float, float, float, float]

    @property
    def half_height(self) -> float:
        return self.size[1] if self.kind == "cylinder" else self.size[2]


#: The trash. Gripped across, near the jaw tips: the open claw is 75 mm across
#: at the tips and 58 mm 55 mm in from the hinge (`five_foot/tools/claw_mouth.py`),
#: so a 66 mm can does not go in (grasp_check, 2026-10-27); these are 48-52 mm.
TRASH: tuple[Trash, ...] = (
    # A slim 250 ml can, flooded: aluminium and the sand it took on.
    Trash("can", "cylinder", (0.020, 0.060), 1300.0, (0.85, 0.08, 0.08, 1.0)),
    # A small glass bottle on its base, flooded: glass (2500) and water averaged.
    Trash("bottle", "cylinder", (0.020, 0.080), 1500.0, (0.10, 0.75, 0.30, 1.0)),
    # A waterlogged drink carton.
    Trash("carton", "box", (0.019, 0.019, 0.055), 1400.0, (0.95, 0.80, 0.05, 1.0)),
)

def _register_heights() -> None:
    from oceanking.demo import perception

    for t in TRASH:
        perception.HEIGHTS[t.name] = 2 * t.half_height


_register_heights()

#: Where things lie, metres from the robot's spawn (origin, facing +x).
LAYOUT: dict[str, tuple[float, float]] = {
    "can": (0.90, 0.35),
    "bottle": (1.30, -0.45),
    "carton": (0.70, -0.85),
}
#: The drop zone: its centre and radius.
DROP_ZONE: tuple[float, float] = (-0.30, 1.00)
DROP_RADIUS = 0.18
ZONE_RGBA = (0.95, 0.10, 0.85, 1.0)


def _trash_spec(t: Trash) -> Any:
    from tasks.jumper.five_foot import objects

    spec = objects._prop_spec()
    body = objects._free_body(spec, t.name)
    if t.kind == "cylinder":
        objects._prop_geom(body, name=t.name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[t.size[0], t.size[1], 0.0], density=t.density,
                           rgba=list(t.rgba))
    else:
        objects._prop_geom(body, name=t.name, type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=list(t.size), density=t.density, rgba=list(t.rgba))
    return spec


def _zone_spec(pos: tuple[float, float]) -> Any:
    # A fixed entity is placed by its own body pose: its `init_state` is a free
    # body's reset target and left this marker at the origin (2026-10-27).
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    body = spec.worldbody.add_body(name="drop_zone", pos=[pos[0], pos[1], 0.0])
    look = dict(contype=0, conaffinity=0, group=2, rgba=list(ZONE_RGBA))
    body.add_geom(name="drop_zone_disk", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[DROP_RADIUS, 0.002, 0.0], pos=[0.0, 0.0, 0.002], **look)
    body.add_geom(name="drop_zone_pole", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[0.012, 0.20, 0.0], pos=[0.0, 0.0, 0.20], **look)
    body.add_geom(name="drop_zone_flag", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[0.005, 0.06, 0.04], pos=[0.0, 0.06, 0.36], **look)
    return spec


def add_demo_scene(cfg: Any, hydro: Any, layout: dict | None = None) -> dict:
    """Add the trash, under water, and the drop zone to `cfg`; pin the robot's spawn
    to the origin facing +x. Returns {trash name: gravcomp}."""
    from functools import partial

    from mjlab.entity import EntityCfg

    from oceanking.props import make_props_wet
    from tasks.jumper.five_foot import objects

    layout = layout or LAYOUT
    entities = cfg.scene.entities
    for t in TRASH:
        x, y = layout[t.name]
        entities[t.name] = EntityCfg(
            spec_fn=partial(_trash_spec, t),
            init_state=EntityCfg.InitialStateCfg(pos=(x, y, t.half_height + 0.002)))
    entities["drop_zone"] = EntityCfg(spec_fn=partial(_zone_spec, DROP_ZONE))
    objects._place_props(cfg, {t.name: None for t in TRASH})
    # The row's contact model, for every contact (as `--objects` sets it).
    cfg.sim.nconmax = max(int(cfg.sim.nconmax or 0), objects.OBJECTS_NCONMAX)
    cfg.sim.njmax = max(int(cfg.sim.njmax or 0), objects.OBJECTS_NJMAX)
    cfg.sim.mujoco.cone = objects.PROP_CONE
    cfg.sim.mujoco.impratio = objects.PROP_IMPRATIO
    objects._place_the_robot(cfg)
    return make_props_wet(cfg, hydro, names=tuple(t.name for t in TRASH))


__all__ = ["DROP_RADIUS", "DROP_ZONE", "LAYOUT", "TRASH", "Trash", "add_demo_scene"]
