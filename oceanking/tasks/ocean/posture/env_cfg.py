"""ocean.posture's environment: `jumper.posture`'s, with water installed."""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from oceanking.buoyancy import check_model
from oceanking.hydro import HydroCfg, install

from ...jumper.common.assets import JUMPER_XML
from ...jumper.posture.env_cfg import env_cfg as posture_env_cfg
from ..common import add_trajectory, make_nominal


def env_cfg(
    asset: Path | None = None,
    play: bool = False,
    hydro_mode: str = "full",
    buoyancy_ratio: float = 0.8,
    buoyancy_model: str = "archimedes",
    crown_air: float = 0.5,
    current: tuple[float, float] = (0.0, 0.0),
    drag_scale: float = 1.0,
    added_mass_coef: float = 1.0,
    nominal: bool = False,
    trajectory_out: str | None = None,
) -> ManagerBasedRlEnvCfg:
    cfg = posture_env_cfg(asset=asset, play=play)
    hydro = HydroCfg(
        mode=hydro_mode,
        buoyancy_ratio=buoyancy_ratio,
        buoyancy_model=buoyancy_model,
        crown_air=crown_air,
        current=(float(current[0]), float(current[1]), 0.0),
        drag_scale=drag_scale,
        added_mass_coef=added_mass_coef,
    )
    if hydro.buoyancy_on and hydro.buoyancy_model == "archimedes":
        check_model(asset or JUMPER_XML)
    install(cfg, hydro)
    print(f"[oceanking] {hydro.describe()}")
    if nominal:
        make_nominal(cfg, play)
    if trajectory_out:
        add_trajectory(cfg, play, trajectory_out, {
            "task": "ocean.posture", "hydro": hydro.as_dict(), "summary": hydro.describe()})
    return cfg
