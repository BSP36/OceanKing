"""ocean.posture's environment: `jumper.posture`'s, with water installed."""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from oceanking.buoyancy import check_model
from oceanking.hydro import HydroCfg, install

from ...jumper.common.assets import JUMPER_XML
from ...jumper.posture.env_cfg import env_cfg as posture_env_cfg


def env_cfg(
    asset: Path | None = None,
    play: bool = False,
    hydro_mode: str = "full",
    buoyancy_ratio: float = 0.8,
    buoyancy_model: str = "archimedes",
    crown_air: float = 0.5,
    current: tuple[float, float] = (0.0, 0.0),
    drag_scale: float = 1.0,
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
    )
    if hydro.buoyancy_on and hydro.buoyancy_model == "archimedes":
        check_model(asset or JUMPER_XML)
    install(cfg, hydro)
    print(f"[oceanking] {hydro.describe()}")
    if nominal:
        _make_nominal(cfg)
    if trajectory_out:
        if not play:
            raise ValueError("--trajectory-out is for replays; training does not record")
        from mjlab.managers.event_manager import EventTermCfg

        from oceanking.trajectory import TrajectoryRecorder

        cfg.events["oceanking_trajectory"] = EventTermCfg(
            func=TrajectoryRecorder,
            mode="step",
            params={
                "path": trajectory_out,
                "meta": {"task": "ocean.posture", "hydro": hydro.as_dict(),
                         "summary": hydro.describe()},
            },
        )
    return cfg


#: Startup randomisation `--nominal` removes, leaving the asset's own values.
_RANDOMISATION = ("foot_friction", "encoder_bias", "base_com")
#: Fixed spawn height above the ground, m (mid of upstream's 0.01-0.05 range).
_NOMINAL_SPAWN_Z = 0.03


def _make_nominal(cfg: ManagerBasedRlEnvCfg) -> None:
    """One robot, one start: what a side-by-side comparison of conditions needs.

    Measured 2026-10-24: with upstream's replay randomisation left on, the same
    command script at beta 0.8 + 0.15 m/s cross-current walked 0.19 m/s in one
    run and 0.32 m/s in the next.
    """
    for name in _RANDOMISATION:
        cfg.events.pop(name, None)
    reset = cfg.events["reset_base"].params
    reset["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0),
                           "z": (_NOMINAL_SPAWN_Z, _NOMINAL_SPAWN_Z), "yaw": (0.0, 0.0)}
    reset["velocity_range"] = {}
