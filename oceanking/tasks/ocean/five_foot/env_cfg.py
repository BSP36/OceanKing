"""ocean.five_foot's environment: `jumper.five_foot`'s, with water installed."""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from oceanking.buoyancy import check_model
from oceanking.hydro import HydroCfg, install

from ...jumper.common.assets import JUMPER_XML
from ...jumper.five_foot.env_cfg import env_cfg as base_env_cfg
from ..common import add_trajectory

#: The arm held out at the grasp preset instead of stowed (docs/GRASP.md): the
#: official policy cannot walk with it out (0 mm in 1.5 s of 0.15 m/s), so a
#: policy is fine-tuned here to walk with the open claw out in front, which is
#: what lets trash enter the mouth from the front.
REACH = {"LF_J0_joint": -1.1013, "LF_J1_joint": -1.3142, "LF_J2_joint": -0.8291,
         "LF_J3_joint": 0.6949}
ARM_BOX = 0.10


#: `--arm reach_lift`: the shoulder (J1) may also be up to 0.5 rad raised from
#: `reach`, which lifts the mouth ~3.5 cm. Within reach's own +-0.1 box the mouth
#: rises at most 9 mm, and a piece held that low drags on the floor and anchors the
#: robot (4 s of 0.15 m/s walked 7-50 mm holding one, 2026-10-27).
LIFT_J1 = 0.5


def hold_arm_at(cfg, pose: dict[str, float], j1_up: float = 0.0) -> None:
    """Hold the carried arm at `pose` (+-ARM_BOX per episode in training; the
    shoulder also up to `j1_up` rad raised) and make it the arm's default (so
    `joint_pos` is observed relative to it, as for the stow)."""
    robot = cfg.scene.entities["robot"]
    robot.init_state.joint_pos = dict(robot.init_state.joint_pos) | dict(pose)
    hold = cfg.events["hold_carried_arm"]
    ranges = list(hold.params["ranges"])
    for k, joint in enumerate(pose):
        lo, hi = ranges[k]
        width = (hi - lo) / 2
        low = pose[joint] - width - (j1_up if joint.endswith("_J1_joint") else 0.0)
        ranges[k] = (low, pose[joint] + width)
    hold.params["ranges"] = tuple(ranges)


def env_cfg(
    asset: Path | None = None,
    play: bool = False,
    objects: bool = False,
    hold: str | None = None,
    buoyancy_ratio: float = 0.8,
    current: tuple[float, float] = (0.0, 0.0),
    trajectory_out: str | None = None,
    demo_scene: bool | str = False,
    arm: str = "stow",
) -> ManagerBasedRlEnvCfg:
    cfg = base_env_cfg(asset=asset, play=play, objects=objects, hold=hold)
    if arm == "reach":
        hold_arm_at(cfg, REACH)
    elif arm == "reach_lift":
        hold_arm_at(cfg, REACH, j1_up=LIFT_J1)
    elif arm != "stow":
        raise ValueError(f"arm must be stow, reach or reach_lift, got {arm!r}")
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

        add_demo_scene(cfg, hydro, demo_scene if isinstance(demo_scene, str) else "full")
    print(f"[oceanking] ocean.five_foot {hydro.describe()}")
    if trajectory_out:
        add_trajectory(cfg, play, trajectory_out, {
            "task": "ocean.five_foot", "hydro": hydro.as_dict()})
    return cfg
