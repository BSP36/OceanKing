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
    kind: str  # "cylinder" (standing), "box", or "bottle" (a wide body and a neck)
    size: tuple[float, ...]  # cylinder (radius, half-height); box half-extents;
    #                          bottle (body radius, body height, neck radius, neck height)
    density: float  # effective, kg/m^3
    rgba: tuple[float, float, float, float]

    @property
    def half_height(self) -> float:
        if self.kind == "bottle":
            return (self.size[1] + self.size[3]) / 2
        return self.size[1] if self.kind == "cylinder" else self.size[2]


#: The trash. Gripped across, near the jaw tips: the open claw is 75 mm across
#: at the tips and 58 mm 55 mm in from the hinge (`five_foot/tools/claw_mouth.py`),
#: so a 66 mm can does not go in (grasp_check, 2026-10-27); these are 38-40 mm.
TRASH: tuple[Trash, ...] = (
    # Two bottles on their bases, silted up: a wide body the claw passes over and a
    # neck it closes on. The wide body is what keeps them standing: slim 10-13 cm
    # cylinders were knocked over by a jaw or a leg on the way in, and a piece
    # lying on its side is out of the claw's reach for good (its mouth rides 7-8 cm
    # up), which ended the first run with Claude (docs/DEMO.md).
    Trash("red_bottle", "bottle", (0.035, 0.050, 0.018, 0.080), 2500.0,
          (0.85, 0.08, 0.08, 1.0)),
    Trash("green_bottle", "bottle", (0.035, 0.050, 0.018, 0.080), 2500.0,
          (0.10, 0.75, 0.30, 1.0)),
    # A drink carton full of mud.
    Trash("carton", "box", (0.019, 0.019, 0.050), 2200.0, (0.95, 0.80, 0.05, 1.0)),
)

def _register_heights() -> None:
    from oceanking.demo import perception

    for t in TRASH:
        perception.HEIGHTS[t.name] = 2 * t.half_height


_register_heights()

#: Where things lie, metres from the robot's spawn (origin, facing +x).
LAYOUT: dict[str, tuple[float, float]] = {
    "red_bottle": (0.90, 0.35),
    "green_bottle": (1.30, -0.45),
    "carton": (0.70, -0.85),
}
#: The drop zone: its centre and radius.
DROP_ZONE: tuple[float, float] = (-0.30, 1.00)
#: Named scenes: which trash lies where, and where the zone is. `full` is the three
#: pieces around the robot; `showcase`, for the submitted video, two pieces in front
#: and the zone a turn to the left (docs/DEMO.md).
LAYOUTS: dict[str, dict] = {
    "full": {"trash": LAYOUT, "zone": DROP_ZONE},
    "showcase": {"trash": {"red_bottle": (0.75, 0.25), "green_bottle": (1.00, -0.20)},
                 "zone": (0.45, 0.95)},
}
DROP_RADIUS = 0.18
ZONE_RGBA = (0.95, 0.10, 0.85, 1.0)


#: Sliding friction of the trash: upstream's row's 0.35. 0.8 (fouled trash) was
#: tried and grasped 2 of 9 against 3-6 of 9 at 0.35 (2026-10-27, small samples).
TRASH_FRICTION = (0.35, 0.0002, 0.00012)


def _trash_spec(t: Trash) -> Any:
    from tasks.jumper.five_foot import objects

    spec = objects._prop_spec()
    body = objects._free_body(spec, t.name)
    look = dict(density=t.density, rgba=list(t.rgba), friction=list(TRASH_FRICTION))
    if t.kind == "bottle":
        r_body, h_body, r_neck, h_neck = t.size
        bottom = -(h_body + h_neck) / 2
        objects._prop_geom(body, name=t.name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[r_body, h_body / 2, 0.0], pos=[0, 0, bottom + h_body / 2],
                           **look)
        objects._prop_geom(body, name=f"{t.name}_neck", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[r_neck, h_neck / 2, 0.0],
                           pos=[0, 0, bottom + h_body + h_neck / 2], **look)
    elif t.kind == "cylinder":
        objects._prop_geom(body, name=t.name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[t.size[0], t.size[1], 0.0], **look)
    else:
        objects._prop_geom(body, name=t.name, type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=list(t.size), **look)
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


def add_demo_scene(cfg: Any, hydro: Any, layout: str = "full") -> dict:
    """Add the trash of `LAYOUTS[layout]`, under water, and its drop zone to `cfg`;
    pin the robot's spawn to the origin facing +x. Returns {trash name: gravcomp}."""
    from functools import partial

    from mjlab.entity import EntityCfg

    from oceanking.props import make_props_wet
    from tasks.jumper.five_foot import objects

    spec = LAYOUTS[layout]
    pieces = [t for t in TRASH if t.name in spec["trash"]]
    entities = cfg.scene.entities
    for t in pieces:
        x, y = spec["trash"][t.name]
        entities[t.name] = EntityCfg(
            spec_fn=partial(_trash_spec, t),
            init_state=EntityCfg.InitialStateCfg(pos=(x, y, t.half_height + 0.002)))
    entities["drop_zone"] = EntityCfg(spec_fn=partial(_zone_spec, spec["zone"]))
    objects._place_props(cfg, {t.name: None for t in pieces})
    # The row's contact model, for every contact (as `--objects` sets it).
    cfg.sim.nconmax = max(int(cfg.sim.nconmax or 0), objects.OBJECTS_NCONMAX)
    cfg.sim.njmax = max(int(cfg.sim.njmax or 0), objects.OBJECTS_NJMAX)
    cfg.sim.mujoco.cone = objects.PROP_CONE
    cfg.sim.mujoco.impratio = objects.PROP_IMPRATIO
    objects._place_the_robot(cfg)
    return make_props_wet(cfg, hydro, names=tuple(t.name for t in pieces))


__all__ = ["DROP_RADIUS", "DROP_ZONE", "LAYOUT", "LAYOUTS", "TRASH", "Trash", "add_demo_scene"]
