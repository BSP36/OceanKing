"""ocean.five_foot's environment: `jumper.five_foot`'s, with water installed."""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from oceanking.buoyancy import check_model
from oceanking.hydro import HydroCfg, install

from ...jumper.common.assets import JUMPER_XML
from ...jumper.five_foot.env_cfg import env_cfg as base_env_cfg
from ..common import add_trajectory


def env_cfg(
    asset: Path | None = None,
    play: bool = False,
    objects: bool = False,
    hold: str | None = None,
    buoyancy_ratio: float = 0.8,
    current: tuple[float, float] = (0.0, 0.0),
    trajectory_out: str | None = None,
    demo_scene: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = base_env_cfg(asset=asset, play=play, objects=objects, hold=hold)
    hydro = HydroCfg(mode="full", buoyancy_ratio=buoyancy_ratio, crown_air=0.5,
                     current=(float(current[0]), float(current[1]), 0.0))
    check_model(asset or JUMPER_XML)
    install(cfg, hydro)
    if objects:
        from oceanking.props import make_props_wet

        make_props_wet(cfg, hydro)
    if demo_scene:
        if not play:
            raise ValueError("--demo-scene is for replays and the demo agent")
        from oceanking.demo.scene import add_demo_scene

        add_demo_scene(cfg, hydro)
    print(f"[oceanking] ocean.five_foot {hydro.describe()}")
    if trajectory_out:
        add_trajectory(cfg, play, trajectory_out, {
            "task": "ocean.five_foot", "hydro": hydro.as_dict()})
    return cfg
