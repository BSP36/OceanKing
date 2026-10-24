"""ocean.posture's environment: `jumper.posture`'s, with water installed."""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from oceanking.hydro import HydroCfg, install

from ...jumper.posture.env_cfg import env_cfg as posture_env_cfg


def env_cfg(
    asset: Path | None = None,
    play: bool = False,
    hydro_mode: str = "full",
    buoyancy_ratio: float = 0.5,
    current: tuple[float, float] = (0.0, 0.0),
    drag_scale: float = 1.0,
) -> ManagerBasedRlEnvCfg:
    cfg = posture_env_cfg(asset=asset, play=play)
    hydro = HydroCfg(
        mode=hydro_mode,
        buoyancy_ratio=buoyancy_ratio,
        current=(float(current[0]), float(current[1]), 0.0),
        drag_scale=drag_scale,
    )
    install(cfg, hydro)
    print(f"[oceanking] {hydro.describe()}")
    return cfg
