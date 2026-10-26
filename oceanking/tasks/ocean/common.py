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
    if "reset_base" not in cfg.events:
        return  # a task that always starts at its default pose (ocean.jump)
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


#: Key `oceanking.tools.expand_checkpoint` writes into a widened checkpoint: its
#: Adam moments describe another network, so they are not restored.
FRESH_OPTIMIZER_KEY = "oceanking_fresh_optimizer"


def fresh_optimizer_runner(base: type) -> type:
    """`base` (an rsl_rl runner), loading a checkpoint marked by
    `expand_checkpoint` without its optimiser state (a fresh Adam), everything
    else -- networks, iteration, the environment's step counter -- restored."""
    import torch

    class OceanRunner(base):
        def load(self, path: str, load_cfg: dict | None = None, strict: bool = True,
                 map_location: str | None = None) -> dict:
            if load_cfg is None:
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if saved.get(FRESH_OPTIMIZER_KEY):
                    load_cfg = {"actor": True, "critic": True, "optimizer": False,
                                "iteration": True, "rnd": True}
                    print(f"[oceanking] {path}: fresh optimiser (expanded checkpoint)")
            return super().load(path, load_cfg, strict, map_location)

    OceanRunner.__name__ = f"Ocean{base.__name__}"
    return OceanRunner
