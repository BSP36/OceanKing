"""Archimedes buoyancy for Jumper, and the OceanKing ballast kit.

See `docs/PHYSICS.md` for the specification this implements; the numbers there
are produced by this module and `oceanking/tools/link_volumes.py`.

## Per-link buoyancy

Every robot link is a sealed housing displacing its own volume V_i, measured once
from the link's collision hull (`oceanking/data/jumper_link_volumes.json`). Its
buoyant force rho * V_i * g acts at the link's **centre of buoyancy** (the
centroid of that volume), not at its centre of mass. MuJoCo's gravity
compensation acts only at a body's centre of mass, so each link carries a welded
"buoyancy point": a child body of negligible mass ``COB_MASS`` at the centroid
whose compensation cancels its own weight and lifts ``rho * V_i``. The link
itself is compensated for nothing.

## Added mass

A body accelerating in water moves water with it. `add_added_mass` adds
``C_a * rho * V_i`` to each link's mass, scales its inertia by the same factor, and
compensates the extra weight, so contact loads and the static balance are
unchanged and only inertia grows. Isotropic, at the link's centre of mass: an
approximation of an anisotropic, distributed effect (docs/PHYSICS.md).

Sealed, Jumper displaces 3.42 L against 2.543 kg (the upstream model's mass), a
buoyancy-to-weight ratio of 1.35: it floats. Walking on the seabed needs ballast.

## The kit

- **Keel**: a lead plate inside the torso, at its floor. Inside the sealed hull, so
  it adds mass and no volume. It is what makes the robot sink, and it sits low,
  which keeps the centre of mass below the centre of buoyancy. It is also the
  trim: placed fore-aft and sideways so that, at HOME, the whole robot's centre
  of mass lies directly under its centre of buoyancy (no static pitch or roll
  moment).
- **Crown** (`oceanking.crown`): a gold band round a velvet dome, the dome and band
  one rigid sealed ballast tank, on the torso's roof. It
  displaces its outer volume whatever is inside; flooding the tank (air fraction
  0) adds water mass, blowing it (air fraction 1) removes it. It sits high, so its
  buoyancy rights the robot.

The keel mass is solved so that the whole robot has the requested overall ratio
``beta = rho * V_total / M_total`` at the requested crown air fraction.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from oceanking import crown


DATA = Path(__file__).resolve().parent / "data" / "jumper_link_volumes.json"

LEAD_DENSITY = 11340.0  # kg/m^3

#: Keel footprint; the thickness follows from the mass.
#: 110 x 90 mm: inside the base's hull at every beta. 120 x 100 mm put about 7 cm^3
#: of lead corners outside the rounded floor (found in review).
KEEL_FOOTPRINT = (0.11, 0.09)  # m, x by y
#: Height of the keel's underside above the torso's lowest point. The hull's floor
#: is rounded: 3.5 mm up at the centre, 6.5-8.0 mm up under the plate's corners
#: (measured on the base's collision hull); 10 mm keeps every corner inside
#: (`tests/test_buoyancy.py::test_the_keel_lies_inside_the_hull`).
KEEL_CLEARANCE = 0.010  # m

#: Ellipsoid fluid model coefficients for drag-only use: blunt drag, slender drag,
#: angular drag, Kutta lift, Magnus lift. The first three are MuJoCo's defaults;
#: lift is switched off (docs/PHYSICS.md).
DRAG_COEFS = (0.5, 0.25, 1.5, 0.0, 0.0)
NO_FLUID = (0.0, 0.0, 0.0, 0.0, 0.0)

LEAD_GREY = (0.35, 0.36, 0.38, 1.0)

#: The keel plate's collision bit: one no geom's conaffinity has, so it touches
#: nothing, but it counts as colliding and survives the native backend's
#: stripping of non-colliding geoms (without its fluid flag, the keel would fall
#: back to the inertia-box drag model there and nowhere else).
KEEL_CONTYPE = 1 << 30
#: Fluid multiplier of the keel plate: positive (the flag), otherwise nothing.
KEEL_FLUID = 1e-9

#: Mass of each buoyancy point, kg. It only has to be small against its link's
#: mass (the smallest link is 1 g) and large enough to keep the compensation
#: factor ``1 + rho V / COB_MASS`` well inside float32 range (base: 1.2e5).
COB_MASS = 1.0e-5
COB_INERTIA = 1.0e-12  # kg m^2


def crown_outer_volume() -> float:
    return crown.outer_volume()


def crown_tank_volume() -> float:
    return crown.tank_volume()


def crown_shell_mass() -> float:
    return crown.dry_mass()


@lru_cache(maxsize=1)
def link_table() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def check_model(xml: Path) -> None:
    """Refuse a model the volume table was not measured on."""
    table = link_table()
    digest = hashlib.sha256(Path(xml).read_bytes()).hexdigest()
    if digest != table["model_sha256"]:
        raise RuntimeError(
            f"{xml} is not the model {DATA.name} was measured on ({table['model']}). "
            "Re-run `.venv/bin/python -m oceanking.tools.link_volumes`."
        )


@dataclass(frozen=True)
class Ballast:
    """The kit for one condition, and the resulting whole-robot numbers (SI)."""

    rho: float
    beta: float
    crown_air: float
    robot_mass: float
    robot_volume: float
    crown_volume: float
    crown_shell_mass: float
    crown_water_mass: float
    keel_mass: float

    @property
    def crown_mass(self) -> float:
        return self.crown_shell_mass + self.crown_water_mass

    @property
    def total_mass(self) -> float:
        return self.robot_mass + self.crown_mass + self.keel_mass

    @property
    def total_volume(self) -> float:
        return self.robot_volume + self.crown_volume

    def keel_centre_xy(self) -> tuple[float, float]:
        """Keel plate centre (base frame) trimming the robot level at HOME."""
        table = link_table()
        com_r = np.asarray(table["home_centre_of_mass_m"][:2])
        cob_r = np.asarray(table["home_centre_of_buoyancy_m"][:2])
        mount = np.asarray(crown_mount()[:2])  # the crown's centres lie on its axis
        target = (self.robot_volume * cob_r + self.crown_volume * mount) / self.total_volume
        moment = self.total_mass * target - self.robot_mass * com_r - self.crown_mass * mount
        return tuple(moment / self.keel_mass) if self.keel_mass > 0 else tuple(target)

    @property
    def keel_thickness(self) -> float:
        return self.keel_mass / (LEAD_DENSITY * KEEL_FOOTPRINT[0] * KEEL_FOOTPRINT[1])

    def beta_at(self, crown_air: float) -> float:
        """Overall ratio with this keel and another crown air fraction."""
        water = self.rho * crown_tank_volume() * (1.0 - crown_air)
        mass = self.robot_mass + self.crown_shell_mass + water + self.keel_mass
        return self.rho * self.total_volume / mass

    def describe(self) -> str:
        return (
            f"archimedes beta={self.beta:g} (robot {self.robot_mass:.3f} kg / "
            f"{self.robot_volume * 1e3:.2f} L, keel {self.keel_mass:.3f} kg lead, "
            f"crown {self.crown_volume * 1e3:.2f} L air={self.crown_air:g}, total "
            f"{self.total_mass:.3f} kg; trim range beta {self.beta_at(0.0):.3f}-"
            f"{self.beta_at(1.0):.3f})"
        )


def solve_ballast(beta: float, crown_air: float, rho: float) -> Ballast:
    """The keel mass giving overall ratio `beta` at crown air fraction `crown_air`."""
    if not 0.0 <= crown_air <= 1.0:
        raise ValueError(f"crown air fraction must be in [0, 1], got {crown_air}")
    if beta <= 0.0:
        raise ValueError(f"archimedes buoyancy needs beta > 0, got {beta}")
    table = link_table()
    robot_mass = table["total_mass_kg"]
    robot_volume = table["total_volume_m3"]
    water = rho * crown_tank_volume() * (1.0 - crown_air)
    keel = rho * (robot_volume + crown_outer_volume()) / beta - (
        robot_mass + crown_shell_mass() + water
    )
    if keel < 0.0:
        light = rho * (robot_volume + crown_outer_volume()) / (
            robot_mass + crown_shell_mass() + water
        )
        raise ValueError(
            f"beta={beta} at crown air {crown_air} needs negative ballast: without a "
            f"keel the robot already has beta={light:.3f}"
        )
    ballast = Ballast(
        rho=rho, beta=beta, crown_air=crown_air, robot_mass=robot_mass,
        robot_volume=robot_volume, crown_volume=crown_outer_volume(),
        crown_shell_mass=crown_shell_mass(), crown_water_mass=water, keel_mass=keel,
    )
    bounds = table["torso_bounds_m"]
    room = 0.5 * (bounds["max"][2] - bounds["min"][2])
    if ballast.keel_thickness > room:
        raise ValueError(
            f"beta={beta} needs {keel:.2f} kg of lead, a {ballast.keel_thickness * 1e3:.0f} mm "
            f"plate: more than half the torso's height"
        )
    x, y = ballast.keel_centre_xy()
    half = np.asarray(KEEL_FOOTPRINT) / 2
    if (x - half[0] < bounds["min"][0] or x + half[0] > bounds["max"][0]
            or y - half[1] < bounds["min"][1] or y + half[1] > bounds["max"][1]):
        raise ValueError(
            f"beta={beta}: the trimming keel would sit at ({x * 1e3:.0f}, {y * 1e3:.0f}) mm, "
            "outside the torso"
        )
    return ballast


def crown_mount() -> tuple[float, float, float]:
    """Where the crown's base sits, base frame: centred on the torso, on its roof."""
    bounds = link_table()["torso_bounds_m"]
    return (0.5 * (bounds["min"][0] + bounds["max"][0]),
            0.5 * (bounds["min"][1] + bounds["max"][1]),
            bounds["max"][2] - crown.SEAT_DEPTH)


# ── spec edits ──────────────────────────────────────────────────────────────


def _link(body: Any) -> str:
    return body.name.rsplit("/", 1)[-1]


def _box_inertia(mass: float, size: tuple[float, float, float]) -> list[float]:
    x, y, z = size
    return [mass * (y * y + z * z) / 12, mass * (x * x + z * z) / 12, mass * (x * x + y * y) / 12]


def _find_links(spec: Any) -> dict[str, Any]:
    links = link_table()["links"]
    found = {_link(body): body for body in spec.bodies if _link(body) in links}
    if set(found) != set(links):
        missing = sorted(set(links) - set(found))
        raise RuntimeError(f"Jumper links not in the model: {missing}")
    return found


def _add_buoyancy_point(parent: Any, pos: list[float], lift_kg: float) -> Any:
    """A welded child at `pos` (parent frame) lifting `lift_kg` * g upward.

    It has no geom, so MuJoCo's inertia-box drag model applies to it: with
    COB_MASS and COB_INERTIA the box is 0.8 mm across, and the 41 points together
    feel ~1e-2 N at 1 m/s against the robot's ~25 N (pinned by a test). A
    zero-coefficient fluid geom would remove even that, at +19% per training
    iteration (measured 2026-10-25: 2.38 s against 2.00 s, 4096 environments)."""
    point = parent.add_body(name=f"{parent.name}_cob", pos=list(pos))
    point.explicitinertial = True
    # MjSpec's default inertial frame is not the body frame: left unset, the
    # compiled ipos equals `pos` and the point sits at twice the intended offset.
    point.ipos = [0.0, 0.0, 0.0]
    point.iquat = [1.0, 0.0, 0.0, 0.0]
    point.mass = COB_MASS
    point.inertia = [COB_INERTIA] * 3
    point.gravcomp = 1.0 + lift_kg / COB_MASS
    return point


def install_on_spec(spec: Any, ballast: Ballast) -> None:
    """Per-link buoyancy at each centre of buoyancy, plus the kit."""
    import mujoco

    links = link_table()["links"]
    found = _find_links(spec)
    for name, body in found.items():
        entry = links[name]
        if not math.isclose(body.mass, entry["mass_kg"], rel_tol=1e-6, abs_tol=1e-9):
            raise RuntimeError(
                f"{body.name} has mass {body.mass} kg, the volume table says "
                f"{entry['mass_kg']} kg: the table is for another model"
            )
        body.gravcomp = 0.0
        _add_buoyancy_point(
            body, entry["centre_of_buoyancy_body_m"], ballast.rho * entry["volume_m3"]
        )

    base = found["base_link"]
    bounds = link_table()["torso_bounds_m"]
    prefix = base.name[: -len("base_link")]

    # Keel: inside the hull, so no volume, no drag and no collision of its own --
    # anything reaching it meets the torso's hull first.
    if ballast.keel_mass > 0.0:
        t = ballast.keel_thickness
        size = (*KEEL_FOOTPRINT, t)
        kx, ky = ballast.keel_centre_xy()
        keel = base.add_body(
            name=f"{prefix}oceanking_keel",
            pos=[kx, ky, bounds["min"][2] + KEEL_CLEARANCE + t / 2],
        )
        keel.explicitinertial = True
        keel.ipos = [0.0, 0.0, 0.0]  # see _add_buoyancy_point
        keel.iquat = [1.0, 0.0, 0.0, 0.0]
        keel.mass = ballast.keel_mass
        keel.inertia = _box_inertia(ballast.keel_mass, size)
        keel.gravcomp = 0.0
        plate = keel.add_geom(
            name=f"{prefix}oceanking_keel_plate", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[s / 2 for s in size], rgba=list(LEAD_GREY), contype=KEEL_CONTYPE,
            conaffinity=0, group=2, mass=0.0,
        )
        # A massive body with no flagged geom falls back to MuJoCo's inertia-box
        # drag. Flagged, with zero coefficients and a negligible multiplier: the
        # multiplier also scales the ellipsoid model's added-mass terms, which the
        # coefficients do not gate, and inside the hull the keel has none.
        plate.fluid_ellipsoid = KEEL_FLUID
        plate.fluid_coefs = list(NO_FLUID)

    # Crown: displaces its outer volume, carries drag, buoyant and high.
    body = base.add_body(name=f"{prefix}oceanking_crown", pos=list(crown_mount()))
    body.explicitinertial = True
    body.mass = ballast.crown_mass
    body.ipos = [0.0, 0.0, crown.centre_height(ballast.crown_water_mass, ballast.rho)]
    body.inertia = crown.inertia(ballast.crown_mass)
    # Its buoyancy acts at its volume's centroid, not at its centre of mass, which
    # the settled tank water pulls down (found when BG dropped with that change).
    body.gravcomp = 0.0
    _add_buoyancy_point(body, [0.0, 0.0, crown.buoyancy_centre_height()],
                        ballast.rho * ballast.crown_volume)
    # Its colliding parts collide the way the torso does (see `_torso_collision`):
    # with the ground and props, not with the robot's own limbs.
    crown.build(spec, body, prefix, _torso_collision(base, spec), DRAG_COEFS)


def _torso_collision(base: Any, spec: Any) -> tuple[int, int, int]:
    """The torso's collision bits, read off its colliding geom rather than copied:
    upstream assigns them per entity, and a hard-coded copy would drift silently."""
    for geom in spec.geoms:
        if geom.parent.name == base.name and (geom.contype or geom.conaffinity):
            return int(geom.contype), int(geom.conaffinity), int(geom.condim)
    raise RuntimeError(f"{base.name} has no colliding geom to copy collision bits from")


#: Added-mass coefficient C_a per drag group (`oceanking.drag.group_of`), before
#: the `added_mass_coef` multiplier. Torso and crown: compact bodies, the sphere's
#: 0.5. Limbs: cylinders moving across their axis, 1.0 -- legs swing broadside.
ADDED_MASS_CA = {"torso": 0.5, "crown": 0.5, "legs": 1.0, "arms": 1.0}


def add_added_mass(spec: Any, scale: float, rho: float) -> None:
    """Add ``scale * C_a * rho * V`` to every Jumper link and to the crown, if
    present, with C_a from `ADDED_MASS_CA` by drag group."""
    from oceanking.drag import group_of

    links = link_table()["links"]
    for name, body in _find_links(spec).items():
        ca = ADDED_MASS_CA[group_of(name)]
        _add_mass(body, scale * ca * rho * links[name]["volume_m3"])
    for body in spec.bodies:
        if _link(body) == "oceanking_crown":
            _add_mass(body, scale * ADDED_MASS_CA["crown"] * rho * crown.outer_volume())


def _add_mass(body: Any, dm: float) -> None:
    """Grow inertia by `dm` and compensate its weight: static forces unchanged."""
    m = body.mass
    if m <= 0.0 or dm <= 0.0:
        return
    body.gravcomp = (body.gravcomp * m + dm) / (m + dm)
    body.inertia = [i * (m + dm) / m for i in body.inertia]
    body.mass = m + dm


def net_weight(model: Any, base: str = "robot/base_link") -> float:
    """Weight minus buoyancy of everything under `base`'s root, in newtons, from a
    compiled model: sum(m_i (1 - gravcomp_i)) g. Added mass and buoyancy points
    contribute nothing to it by construction."""
    tree = model.body_rootid == model.body_rootid[model.body(base).id]
    m = model.body_mass[tree]
    return float((m * (1.0 - model.body_gravcomp[tree])).sum() * -model.opt.gravity[2])


def set_drag_geoms(spec: Any, factors: dict[str, float] | float) -> None:
    """Geometry-based drag: the ellipsoid fluid model, each geom's force scaled by
    its multiplier (`fluid_ellipsoid`).

    A float flags every colliding geom with that multiplier (MuJoCo's raw model at
    1.0, for bodies other than Jumper). A dict maps Jumper's drag groups
    (`oceanking.drag.group_of`) to their calibrated multipliers; then only the
    robot's colliding geoms and the crown's flagged geoms are set, and a colliding
    robot geom outside every group raises."""
    from oceanking.drag import group_of

    if not isinstance(factors, dict):
        for geom in spec.geoms:
            if geom.contype or geom.conaffinity:
                geom.fluid_ellipsoid = float(factors)
                geom.fluid_coefs = list(DRAG_COEFS)
        return
    links = set(link_table()["links"])
    for geom in spec.geoms:
        link = _link(geom.parent)
        if link == "oceanking_crown":
            if geom.fluid_ellipsoid > 0:
                geom.fluid_ellipsoid = factors["crown"]
            continue
        if link not in links or not (geom.contype or geom.conaffinity):
            continue
        group = group_of(link)
        if group is None:
            raise RuntimeError(f"{geom.name} on {link} belongs to no drag group")
        geom.fluid_ellipsoid = factors[group]
        geom.fluid_coefs = list(DRAG_COEFS)


__all__ = [
    "Ballast", "add_added_mass", "check_model", "crown_outer_volume", "crown_tank_volume",
    "install_on_spec", "link_table", "net_weight", "set_drag_geoms", "solve_ballast",
]
