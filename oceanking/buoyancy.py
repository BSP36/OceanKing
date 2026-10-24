"""Archimedes buoyancy for Jumper, and the OceanKing ballast kit.

See `docs/PHYSICS.md` for the specification this implements; the numbers there
are produced by this module and `oceanking/tools/link_volumes.py`.

## Per-link buoyancy

Every robot link is a sealed housing displacing its own volume V_i, measured once
from the link's collision hull (`oceanking/data/jumper_link_volumes.json`). Its
buoyant force rho * V_i * g is applied at the link's centre of mass through
MuJoCo's gravity compensation, ``gravcomp_i = rho * V_i / m_i``.

Sealed, Jumper displaces 3.42 L against 2.543 kg (the upstream model's mass), a
buoyancy-to-weight ratio of 1.35: it floats. Walking on the seabed needs ballast.

## The kit

- **Keel**: a lead plate inside the torso, at its floor. Inside the sealed hull, so
  it adds mass and no volume. It is what makes the robot sink, and it sits low,
  which keeps the centre of mass below the centre of buoyancy.
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

from oceanking import crown


DATA = Path(__file__).resolve().parent / "data" / "jumper_link_volumes.json"

LEAD_DENSITY = 11340.0  # kg/m^3

#: Keel footprint; the thickness follows from the mass.
KEEL_FOOTPRINT = (0.12, 0.10)  # m, x by y
#: Gap between the torso's lowest point and the keel's underside.
KEEL_CLEARANCE = 0.005  # m

#: Ellipsoid fluid model coefficients for drag-only use: blunt drag, slender drag,
#: angular drag, Kutta lift, Magnus lift. The first three are MuJoCo's defaults;
#: lift is switched off (docs/PHYSICS.md).
DRAG_COEFS = (0.5, 0.25, 1.5, 0.0, 0.0)
NO_FLUID = (0.0, 0.0, 0.0, 0.0, 0.0)

LEAD_GREY = (0.35, 0.36, 0.38, 1.0)


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
    return ballast


# ── spec edits ──────────────────────────────────────────────────────────────


def _link(body: Any) -> str:
    return body.name.rsplit("/", 1)[-1]


def _box_inertia(mass: float, size: tuple[float, float, float]) -> list[float]:
    x, y, z = size
    return [mass * (y * y + z * z) / 12, mass * (x * x + z * z) / 12, mass * (x * x + y * y) / 12]


def install_on_spec(spec: Any, ballast: Ballast) -> None:
    """Per-link buoyancy plus the kit, on a spec holding Jumper's links."""
    import mujoco

    links = link_table()["links"]
    found = {}
    for body in spec.bodies:
        if _link(body) in links:
            found[_link(body)] = body
    if set(found) != set(links):
        missing = sorted(set(links) - set(found))
        raise RuntimeError(f"archimedes buoyancy: Jumper links not in the model: {missing}")

    for name, body in found.items():
        entry = links[name]
        if not math.isclose(body.mass, entry["mass_kg"], rel_tol=1e-6, abs_tol=1e-9):
            raise RuntimeError(
                f"{body.name} has mass {body.mass} kg, the volume table says "
                f"{entry['mass_kg']} kg: the table is for another model"
            )
        body.gravcomp = ballast.rho * entry["volume_m3"] / body.mass if body.mass > 0 else 0.0

    base = found["base_link"]
    bounds = link_table()["torso_bounds_m"]
    cx = 0.5 * (bounds["min"][0] + bounds["max"][0])
    cy = 0.5 * (bounds["min"][1] + bounds["max"][1])
    prefix = base.name[: -len("base_link")]

    # Keel: inside the hull, so no volume, no drag and no collision of its own --
    # anything reaching it meets the torso's hull first.
    if ballast.keel_mass > 0.0:
        t = ballast.keel_thickness
        size = (*KEEL_FOOTPRINT, t)
        keel = base.add_body(
            name=f"{prefix}oceanking_keel",
            pos=[cx, cy, bounds["min"][2] + KEEL_CLEARANCE + t / 2],
        )
        keel.explicitinertial = True
        keel.mass = ballast.keel_mass
        keel.inertia = _box_inertia(ballast.keel_mass, size)
        keel.gravcomp = 0.0
        g = keel.add_geom(
            name=f"{prefix}oceanking_keel_plate", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[s / 2 for s in size], rgba=list(LEAD_GREY), contype=0, conaffinity=0,
            group=2, mass=0.0,
        )
        g.fluid_ellipsoid = 1.0
        g.fluid_coefs = list(NO_FLUID)

    # Crown: displaces its outer volume, carries drag, buoyant and high.
    body = base.add_body(
        name=f"{prefix}oceanking_crown", pos=[cx, cy, bounds["max"][2] - crown.SEAT_DEPTH]
    )
    body.explicitinertial = True
    body.mass = ballast.crown_mass
    body.ipos = [0.0, 0.0, crown.centre_height()]
    body.inertia = crown.inertia(ballast.crown_mass)
    body.gravcomp = ballast.rho * ballast.crown_volume / ballast.crown_mass
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


def whole_robot_beta(model: Any, base: str = "robot/base_link") -> float:
    """Buoyant force over weight for every body under `base`'s root, from a
    compiled model: sum(m_i * gravcomp_i) / sum(m_i)."""
    tree = model.body_rootid == model.body_rootid[model.body(base).id]
    mass = model.body_mass[tree]
    return float((mass * model.body_gravcomp[tree]).sum() / mass.sum())


def set_drag_geoms(spec: Any) -> None:
    """Geometry-based drag: the ellipsoid fluid model on every colliding geom."""
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.fluid_ellipsoid = 1.0
            geom.fluid_coefs = list(DRAG_COEFS)


__all__ = [
    "Ballast", "check_model", "crown_outer_volume", "crown_tank_volume", "install_on_spec",
    "link_table", "set_drag_geoms", "solve_ballast", "whole_robot_beta",
]
