"""Minimal underwater physics: buoyancy, drag and a uniform current.

The specification is `docs/PHYSICS.md`; this module and `oceanking/buoyancy.py`
implement it. Everything here is a **model-level** edit made on the `MjSpec`
before compilation, so MuJoCo applies it inside every physics substep on both of
Jumper's backends (MuJoCo Warp on the GPU and native MuJoCo on the CPU) with no
Python in the step loop and nothing to clear on reset.

- **Buoyancy** is MuJoCo's gravity compensation, ``gravcomp_i = rho V_i / m_i``:
  an upward force ``rho V_i g`` at each body's centre of mass. Two models:
  ``archimedes`` (default) takes V_i from each link's measured hull and adds the
  OceanKing ballast kit (`oceanking.buoyancy`); ``uniform`` gives every body the
  same ratio, a diagnostic with no geometry behind it.
- **Drag** is MuJoCo's ellipsoid fluid model on every colliding geom (blunt,
  slender and angular drag; lift terms off), with ``option.wind`` as the current,
  so it acts on the velocity relative to the water and depends on shape, not on
  how mass is distributed inside a body. It contains no buoyancy term, so it never
  double-counts with gravcomp. Summed over overlapping geoms with no shielding it
  overstates drag unevenly, so each part of the robot is calibrated on its own
  (`oceanking.drag`).
- **Added mass** (with drag): ``C_a rho V`` of extra inertia per link, its weight
  compensated (`oceanking.buoyancy.add_added_mass`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

try:  # MuJoCo Warp's fluid torque misses each geom's lever arm; see warp_patch.
    from oceanking import warp_patch as _warp_patch
except ImportError:  # no MuJoCo Warp: native backend only, nothing to fix
    _warp_patch = None
else:
    _warp_patch.install()

Mode = Literal["off", "buoyancy", "drag", "full"]
MODES: tuple[str, ...] = ("off", "buoyancy", "drag", "full")
BuoyancyModel = Literal["archimedes", "uniform"]
BUOYANCY_MODELS: tuple[str, ...] = ("archimedes", "uniform")

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

    buoyancy_ratio: float = 0.8
    """beta = rho V / M, buoyant force over weight. ``archimedes``: of the whole
    robot, reached by sizing the keel. ``uniform``: of every body. 1 is neutral;
    below 1 the robot stays on the seabed."""

    buoyancy_model: BuoyancyModel = "archimedes"
    """How buoyancy is distributed; see the module docstring."""

    crown_air: float = 0.5
    """Air fraction of the crown's ballast tank (``archimedes`` only): 0 flooded,
    1 blown. The keel is sized for `buoyancy_ratio` at this fraction."""

    water_density: float = WATER_DENSITY
    """kg/m^3, used by the drag model."""

    water_viscosity: float = WATER_VISCOSITY
    """Pa*s, the linear (Stokes) part of the drag model."""

    drag_scale: float = 1.0
    """Multiplier on the calibrated drag. 1.0 is the calibration's target drag
    coefficient (docs/PHYSICS.md); use it for sensitivity studies."""

    drag_calibration: float | None = None
    """None: Jumper's per-group drag multipliers, measured
    (`oceanking.drag`, `oceanking/data/jumper_drag.json`). A float: that one
    multiplier on every colliding geom; 1.0 is MuJoCo's raw model, for bodies
    other than Jumper."""

    added_mass_coef: float = 1.0
    """Multiplier on the added mass, on with drag (a hydrodynamic effect): 1.0 is
    C_a 0.5 on the torso and crown and 1.0 on the limbs
    (`oceanking.buoyancy.ADDED_MASS_CA`); 0 switches it off."""

    current: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Uniform fluid velocity in the world frame, m/s."""

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"hydro mode {self.mode!r} is not one of {MODES}")
        if self.buoyancy_ratio < 0.0:
            raise ValueError(f"buoyancy_ratio must be >= 0, got {self.buoyancy_ratio}")
        if self.buoyancy_model not in BUOYANCY_MODELS:
            raise ValueError(
                f"buoyancy model {self.buoyancy_model!r} is not one of {BUOYANCY_MODELS}"
            )
        if not 0.0 <= self.crown_air <= 1.0:
            raise ValueError(f"crown_air must be in [0, 1], got {self.crown_air}")
        if self.water_density < 0.0 or self.water_viscosity < 0.0 or self.drag_scale < 0.0:
            raise ValueError("water_density, water_viscosity and drag_scale must be >= 0")
        if self.added_mass_coef < 0.0:
            raise ValueError(f"added_mass_coef must be >= 0, got {self.added_mass_coef}")
        if self.drag_calibration is not None and self.drag_calibration <= 0.0:
            raise ValueError(f"drag_calibration must be > 0, got {self.drag_calibration}")
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
        if self.buoyancy_on and self.buoyancy_model == "archimedes":
            parts.append(self.ballast().describe())
        elif self.buoyancy_on:
            parts.append(f"uniform beta={self.buoyancy_ratio:g}")
        if self.drag_on:
            cal = self.calibration()
            cal_text = (" ".join(f"{k} {v:.2f}" for k, v in sorted(cal.items()))
                        if isinstance(cal, dict) else f"uniform {cal:g}")
            parts.append(
                f"rho={self.water_density:g} drag x{self.drag_scale:g} "
                f"(calibration {cal_text}) added_mass={self.added_mass_coef:g}"
            )
            parts.append("current=({:g}, {:g}, {:g}) m/s".format(*self.current))
        return " ".join(parts)


    def ballast(self):
        """The kit for this condition (``archimedes`` only)."""
        from oceanking.buoyancy import solve_ballast

        return solve_ballast(self.buoyancy_ratio, self.crown_air, self.water_density)

    def calibration(self) -> dict[str, float] | float:
        if self.drag_calibration is not None:
            return self.drag_calibration
        from oceanking.drag import group_factors

        return group_factors()


def apply_to_spec(spec: Any, cfg: HydroCfg) -> None:
    """Write `cfg` into an `mujoco.MjSpec` before it is compiled.

    ``uniform`` sets every body whose `gravcomp` is still zero; a body that
    declares its own (a prop) keeps it. ``archimedes`` sets Jumper's links and
    adds the kit, and leaves every other body alone.
    """
    if cfg.buoyancy_on and cfg.buoyancy_model == "uniform":
        for body in spec.bodies:
            if body.name == "world" or body.gravcomp != 0.0:
                continue
            body.gravcomp = cfg.buoyancy_ratio
    elif cfg.buoyancy_on:
        from oceanking.buoyancy import install_on_spec

        install_on_spec(spec, cfg.ballast())
    if cfg.drag_on:
        from oceanking.buoyancy import add_added_mass, set_drag_geoms

        set_drag_geoms(spec, cfg.calibration())
        spec.option.density = cfg.water_density * cfg.drag_scale
        spec.option.viscosity = cfg.water_viscosity * cfg.drag_scale
        spec.option.wind = list(cfg.current)
        if cfg.added_mass_coef > 0.0:
            add_added_mass(spec, cfg.added_mass_coef, cfg.water_density)


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
