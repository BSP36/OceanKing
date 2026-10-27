"""Props under water: Archimedes' buoyancy for every free object in a scene.

MuJoCo's fluid model (`opt.density`, the inertia-box model every prop falls back
to) gives drag and nothing else: a box of density 2000 at rest in `density=1000`
water accelerates at the full -9.81 m/s^2 (measured 2026-10-27). So a prop's
buoyancy is added the way the robot's is: as gravity compensation,
``gravcomp = rho_water V / m`` on its body. `V` is the volume of its own collision
shape, taken by compiling the prop a second time with every geom at the water's
density (exact for boxes, cylinders and the convex hulls of `objects.py` alike),
so `gravcomp` is the ratio of the two compiled masses.

A gravcomp below 1 sinks, above 1 floats. The props of upstream's row, under water:
soap 0.93 (sinks), can 1.01 (floats: 330 ml of cola in a sealed can is about as
dense as water), notebook 2.4 (floats). `TRASH` gives the scene things that sink.
"""

from __future__ import annotations

from typing import Any

import mujoco


def displaced_ratio(spec_fn: Any, rho: float = 1000.0) -> dict[str, float]:
    """`rho V / m` for each body of the prop that `spec_fn` builds."""
    dry = spec_fn().compile()
    wet_spec = spec_fn()
    for geom in wet_spec.geoms:
        geom.density = rho
    wet = wet_spec.compile()
    out = {}
    for b in range(1, dry.nbody):
        if dry.body_mass[b] > 0:
            out[dry.body(b).name] = float(wet.body_mass[b] / dry.body_mass[b])
    return out


def wet(spec_fn: Any, rho: float = 1000.0) -> Any:
    """`spec_fn`, its bodies given their buoyancy as gravity compensation."""
    ratios = displaced_ratio(spec_fn, rho)

    def build():
        spec = spec_fn()
        for body in spec.bodies:
            if body.name in ratios:
                body.gravcomp = ratios[body.name]
        return spec

    build.oceanking_buoyancy = ratios
    return build


def make_props_wet(cfg: Any, hydro: Any, names: tuple[str, ...] | None = None) -> dict:
    """Give every prop entity of `cfg` (all but the robot, or `names`) its buoyancy.
    Returns {entity: {body: gravcomp}}."""
    report = {}
    for name, entity in cfg.scene.entities.items():
        if name == "robot" or (names is not None and name not in names):
            continue
        entity.spec_fn = wet(entity.spec_fn, hydro.water_density)
        report[name] = entity.spec_fn.oceanking_buoyancy
    print(f"[oceanking] props under water (gravcomp = rho V / m): {report}")
    return report


__all__ = ["displaced_ratio", "make_props_wet", "wet"]


if __name__ == "__main__":  # pragma: no cover - a quick report
    from tasks.jumper.five_foot import objects

    for name, ent in objects._row().items():
        print(name, displaced_ratio(ent.spec_fn))
    del mujoco
