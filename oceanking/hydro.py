"""Minimal underwater physics: buoyancy, drag and a uniform current.

Everything here is a **model-level** edit made on the `MjSpec` before compilation,
so MuJoCo applies it inside every physics substep on both of Jumper's backends
(MuJoCo Warp on the GPU and native MuJoCo on the CPU) with no Python in the step
loop and nothing to clear on reset.

## Buoyancy: `body.gravcomp`

For a fully submerged rigid body of mass m_i and effective sealed volume V_i, the
buoyant force is ``-rho * V_i * g`` at the centre of buoyancy. With the centre of
buoyancy at the centre of mass (the first approximation), MuJoCo's gravity
compensation is exactly that force::

    gravcomp_i = rho * V_i / m_i

We use one buoyancy-to-weight ratio ``beta`` for every body, i.e. the robot is
treated as having a uniform effective density ``rho / beta``. Gravity, masses and
inertias are left untouched. A body whose `gravcomp` is already non-zero when this
runs (a prop that declares its own buoyancy) is left alone, so nothing is
compensated twice.

Not represented: an offset centre of buoyancy, partial submersion (a water
surface) and added mass. Those need an explicit per-body wrench.

## Drag and current: MuJoCo's inertia-box fluid model

``option.density`` / ``option.viscosity`` switch on MuJoCo's inertia-box fluid
model: per body, quadratic drag ``0.5 * rho * A * |v| v`` on the faces of the
body's equivalent inertia box, a matching quadratic rotational drag, and a
viscous (Stokes) term. ``option.wind`` is the fluid velocity, so drag acts on the
velocity **relative to the current**: a body at rest in a moving fluid is pushed
downstream. The model contains **no buoyancy term** (checked in MuJoCo Warp
3.11.0, `_src/passive.py::_fluid_force`, and by `tests/test_hydro.py`), so it
does not double-count with `gravcomp`.

`drag_scale` multiplies the density seen by the drag model only; buoyancy is set
by `buoyancy_ratio` and does not depend on it.

## Cost

Measured on the RTX PRO 6000 Blackwell / Ryzen 9 9950X3D workstation, 2026-10-24,
`ocean.posture`, 4096 environments, MuJoCo Warp 3.11.0: 1.31 s per PPO iteration
with `--hydro off` and 1.36 s with `--hydro full` (+4%).

## Water depth

With a fully submerged, incompressible model the depth has no effect on these
forces. Depth is therefore a property of how a scene *looks* (light attenuation),
not of this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

Mode = Literal["off", "buoyancy", "drag", "full"]
MODES: tuple[str, ...] = ("off", "buoyancy", "drag", "full")

#: Fresh water at about 20 degC. Declared scenario values, not calibrated ones.
WATER_DENSITY = 1000.0  # kg/m^3
WATER_VISCOSITY = 1.0e-3  # Pa*s


@dataclass(frozen=True)
class HydroCfg:
    """One underwater condition. Every entry point that builds an environment for
    the same experiment must be handed the same instance."""

    mode: Mode = "full"
    """Which effects are on: ``off`` (land), ``buoyancy``, ``drag`` (drag and
    current, no buoyancy) or ``full``."""

    buoyancy_ratio: float = 0.5
    """beta = rho * V / m for every body. 0 is land, 1 is neutral buoyancy; values
    below 1 keep the robot on the seabed."""

    water_density: float = WATER_DENSITY
    """kg/m^3, used by the drag model."""

    water_viscosity: float = WATER_VISCOSITY
    """Pa*s, the linear (Stokes) part of the drag model."""

    drag_scale: float = 1.0
    """Multiplies the density the drag model sees. 1.0 is the inertia-box model
    as MuJoCo defines it."""

    current: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Uniform fluid velocity in the world frame, m/s."""

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"hydro mode {self.mode!r} is not one of {MODES}")
        if self.buoyancy_ratio < 0.0:
            raise ValueError(f"buoyancy_ratio must be >= 0, got {self.buoyancy_ratio}")
        if self.water_density < 0.0 or self.water_viscosity < 0.0 or self.drag_scale < 0.0:
            raise ValueError("water_density, water_viscosity and drag_scale must be >= 0")
        if len(self.current) != 3:
            raise ValueError(f"current must have three components, got {self.current}")

    @property
    def buoyancy_on(self) -> bool:
        return self.mode in ("buoyancy", "full")

    @property
    def drag_on(self) -> bool:
        return self.mode in ("drag", "full")

    def with_(self, **changes: Any) -> HydroCfg:
        return replace(self, **changes)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        if self.mode == "off":
            return "hydro off (land)"
        parts = [f"hydro mode={self.mode}"]
        if self.buoyancy_on:
            parts.append(f"beta={self.buoyancy_ratio:g}")
        if self.drag_on:
            parts.append(f"rho={self.water_density:g} drag_scale={self.drag_scale:g}")
            parts.append("current=({:g}, {:g}, {:g}) m/s".format(*self.current))
        return " ".join(parts)


def apply_to_spec(spec: Any, cfg: HydroCfg) -> None:
    """Write `cfg` into an `mujoco.MjSpec` before it is compiled."""
    if cfg.buoyancy_on:
        for body in spec.bodies:
            if body.name == "world" or body.gravcomp != 0.0:
                continue
            body.gravcomp = cfg.buoyancy_ratio
    if cfg.drag_on:
        spec.option.density = cfg.water_density * cfg.drag_scale
        spec.option.viscosity = cfg.water_viscosity
        spec.option.wind = list(cfg.current)


def install(env_cfg: Any, cfg: HydroCfg) -> None:
    """Install `cfg` on an mjlab environment config through `SceneCfg.spec_fn`.

    A `spec_fn` already present is kept and runs first. A scene applied after
    this (`--scene`) wraps it in turn and runs its own edits last, so the water
    reaches every body the scene's props add as entities.
    """
    previous = env_cfg.scene.spec_fn

    def edit(spec: Any) -> None:
        if previous is not None:
            previous(spec)
        apply_to_spec(spec, cfg)

    env_cfg.scene.spec_fn = edit
