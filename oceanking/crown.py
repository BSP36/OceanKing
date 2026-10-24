"""The OceanKing crown: its shape, and the numbers buoyancy needs from it.

A royal crown is a gold band with points round its rim and a velvet cap inside.
Here the cap is the **ballast tank** (see `oceanking.buoyancy`): a sealed shell
of a cylinder (inside the band) under a half-ellipsoid dome, PETG walls, holding
air or water. On it sit five gold points, an orb on the dome's crest and jewels
set flush into the band.

Every part that stands proud of the band and dome collides (band, dome, points
and their tips, orb), so nothing can pass through the ground; the jewels are flush with the band
and are visual only. All collision geoms take the bits the caller passes (the
torso's).

Frame: the crown body's origin is the centre of the band's base, z up.
"""

from __future__ import annotations

import math
from typing import Any

PETG_DENSITY = 1270.0  # kg/m^3

BAND_RADIUS = 0.065  # m
BAND_HEIGHT = 0.045  # m
#: The dome's base sits just inside the band and rises from the band's top.
DOME_RADIUS = 0.061  # m
DOME_HEIGHT = 0.045  # m, semi-axis
WALL = 0.003  # m, tank shell
POINTS = 5
POINT_HEIGHT = 0.038  # m above the band's top
POINT_WIDTH = 0.040  # m at the base
POINT_THICKNESS = 0.005  # m
POINT_SINK = 0.006  # m the point's base reaches down into the band
ORB_RADIUS = 0.012  # m
#: How far the band's base sits below the torso's highest point. The roof is
#: curved, so a crown resting on its crest shows a gap at the rim; seated 3 mm
#: down it does not. The overlap with the hull is at most pi r^2 * 3 mm = 40 cm^3
#: (1.2% of the robot's volume), counted twice -- an accepted approximation.
SEAT_DEPTH = 0.003  # m

GOLD = (0.95, 0.74, 0.16, 1.0)
VELVET = (0.10, 0.16, 0.58, 1.0)
JEWELS = ((0.85, 0.05, 0.12, 1.0), (0.05, 0.55, 0.25, 1.0), (0.15, 0.35, 0.95, 1.0))


def _tank_outer() -> float:
    band = math.pi * BAND_RADIUS**2 * BAND_HEIGHT
    dome = 2.0 / 3.0 * math.pi * DOME_RADIUS**2 * DOME_HEIGHT
    return band + dome


def tank_volume() -> float:
    """Inside the shell: what the tank can hold."""
    band = math.pi * (BAND_RADIUS - WALL) ** 2 * (BAND_HEIGHT - WALL)
    dome = 2.0 / 3.0 * math.pi * (DOME_RADIUS - WALL) ** 2 * (DOME_HEIGHT - WALL)
    return band + dome


def _point_volume() -> float:
    return 0.5 * POINT_WIDTH * (POINT_HEIGHT + POINT_SINK) * POINT_THICKNESS


def _solid_volume() -> float:
    """Solid PETG outside the tank shell: the points and the orb."""
    return POINTS * _point_volume() + 4.0 / 3.0 * math.pi * ORB_RADIUS**3


def outer_volume() -> float:
    """Water the crown displaces, whatever is in the tank."""
    return _tank_outer() + _solid_volume()


def dry_mass() -> float:
    """PETG shell, points and orb; no water."""
    return (_tank_outer() - tank_volume() + _solid_volume()) * PETG_DENSITY


def centre_height() -> float:
    """Height of the crown's centre of mass above its base, for the inertial.
    Approximated at the tank's centroid (band cylinder plus dome)."""
    band = math.pi * BAND_RADIUS**2 * BAND_HEIGHT
    dome = 2.0 / 3.0 * math.pi * DOME_RADIUS**2 * DOME_HEIGHT
    z_dome = BAND_HEIGHT + 3.0 / 8.0 * DOME_HEIGHT
    return (band * BAND_HEIGHT / 2 + dome * z_dome) / (band + dome)


def inertia(mass: float) -> list[float]:
    """Diagonal inertia about the centre, as a solid cylinder of the whole height."""
    h = BAND_HEIGHT + DOME_HEIGHT
    side = mass * (3 * BAND_RADIUS**2 + h * h) / 12
    return [side, side, mass * BAND_RADIUS**2 / 2]


def _point_mesh(spec: Any, name: str) -> None:
    """A flat triangular plate in the x-z plane, thickness along y, base at z=0."""
    w, h, t = POINT_WIDTH / 2, POINT_HEIGHT + POINT_SINK, POINT_THICKNESS / 2
    verts = [-w, -t, 0, w, -t, 0, 0, -t, h, -w, t, 0, w, t, 0, 0, t, h]
    faces = [0, 1, 2, 3, 5, 4, 0, 3, 4, 0, 4, 1, 1, 4, 5, 1, 5, 2, 2, 5, 3, 2, 3, 0]
    spec.add_mesh(name=name, uservert=verts, userface=faces)


def build(spec: Any, body: Any, prefix: str, collision: tuple[int, int, int],
          fluid_coefs: tuple[float, ...]) -> None:
    """Add the crown's geoms to `body` (already placed and given its inertial)."""
    import mujoco

    contype, conaffinity, condim = collision
    solid = dict(contype=contype, conaffinity=conaffinity, condim=condim, group=2, mass=0.0)
    visual = dict(contype=0, conaffinity=0, group=2, mass=0.0)

    band = body.add_geom(
        name=f"{prefix}oceanking_crown_band", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[BAND_RADIUS, BAND_HEIGHT / 2, 0], pos=[0, 0, BAND_HEIGHT / 2],
        rgba=list(GOLD), **solid,
    )
    dome = body.add_geom(
        name=f"{prefix}oceanking_crown_dome", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        size=[DOME_RADIUS, DOME_RADIUS, DOME_HEIGHT], pos=[0, 0, BAND_HEIGHT],
        rgba=list(VELVET), **solid,
    )
    for g in (band, dome):
        g.fluid_ellipsoid = 1.0
        g.fluid_coefs = list(fluid_coefs)

    mesh = f"{prefix}oceanking_crown_point"
    _point_mesh(spec, mesh)
    z0 = BAND_HEIGHT - POINT_SINK
    for k in range(POINTS):
        a = 2 * math.pi * (k + 0.5) / POINTS
        r = BAND_RADIUS - POINT_THICKNESS / 2
        # The plate's x axis along the band's tangent, its thickness radial.
        quat = [math.cos((a + math.pi / 2) / 2), 0, 0, math.sin((a + math.pi / 2) / 2)]
        body.add_geom(
            name=f"{prefix}oceanking_crown_point{k}", type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh, pos=[r * math.cos(a), r * math.sin(a), z0], quat=quat,
            rgba=list(GOLD), **solid,
        )
        tip = BAND_HEIGHT + POINT_HEIGHT
        body.add_geom(
            name=f"{prefix}oceanking_crown_tip{k}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.0055, 0, 0], pos=[r * math.cos(a), r * math.sin(a), tip],
            rgba=list(GOLD), **solid,
        )
    for k in range(2 * POINTS):
        a = 2 * math.pi * k / (2 * POINTS)
        body.add_geom(
            name=f"{prefix}oceanking_crown_jewel{k}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size=[0.0045, 0.008, 0.008],
            pos=[BAND_RADIUS * math.cos(a), BAND_RADIUS * math.sin(a), BAND_HEIGHT / 2],
            quat=[math.cos(a / 2), 0, 0, math.sin(a / 2)],
            rgba=list(JEWELS[k % len(JEWELS)]), **visual,
        )
    body.add_geom(
        name=f"{prefix}oceanking_crown_orb", type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[ORB_RADIUS, 0, 0], pos=[0, 0, BAND_HEIGHT + DOME_HEIGHT + ORB_RADIUS * 0.6],
        rgba=list(GOLD), **solid,
    )
