"""Replay helpers shared by OceanKing's tasks: the nominal start and the trajectory recorder."""

from __future__ import annotations

from typing import Any

#: Startup randomisation `--nominal` removes, leaving the asset's own values.
RANDOMISATION = ("foot_friction", "encoder_bias", "base_com")
#: Fixed spawn height above the ground, m (mid of upstream's 0.01-0.05 range).
NOMINAL_SPAWN_Z = 0.03


def make_nominal(cfg: Any, play: bool) -> None:
    """One robot, one start: what a side-by-side comparison of conditions needs.

    Measured 2026-10-24: with upstream's replay randomisation left on, the same
    command script at beta 0.8 + 0.15 m/s cross-current walked 0.19 m/s in one
    run and 0.32 m/s in the next.
    """
    if not play:
        raise ValueError(
            "--nominal (or OCEAN_NOMINAL) is for replays: in training it would "
            "silently remove the startup randomisation"
        )
    for name in RANDOMISATION:
        cfg.events.pop(name, None)
    reset = cfg.events["reset_base"].params
    reset["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0),
                           "z": (NOMINAL_SPAWN_Z, NOMINAL_SPAWN_Z), "yaw": (0.0, 0.0)}
    reset["velocity_range"] = {}


def add_trajectory(cfg: Any, play: bool, path: str, meta: dict) -> None:
    """Record root pose and command every control step (`oceanking.trajectory`)."""
    if not play:
        raise ValueError("--trajectory-out is for replays; training does not record")
    from mjlab.managers.event_manager import EventTermCfg

    from oceanking.trajectory import TrajectoryRecorder

    cfg.events["oceanking_trajectory"] = EventTermCfg(
        func=TrajectoryRecorder, mode="step", params={"path": path, "meta": meta},
    )
