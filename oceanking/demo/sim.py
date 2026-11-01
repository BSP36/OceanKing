"""The demo robot in its world: `ocean.five_foot` with the sunken trash, driven by calls.

`OceanSim` builds one environment (native MuJoCo, CPU), loads a five-foot policy and
gives the agent's skills something to call: walk, turn, stand, swing the carried
arm to a preset, open and close the claw, and look through the robot's own
**onboard camera** (`jumper.xml`'s `onboard`, on `camera_link`: 1920 x 1440,
95.7 deg vertical field of view; rendered here smaller). It also exposes the
ground truth (robot pose, where each piece of trash is, what the claw holds) for
the tests and the score, never to the agent.

The policy walks; everything else is written around it each control step:
- the velocity command (`twist`) and the attitude command (`body_pose`: pitch,
  roll, twist) through each term's `compute`, as `mjrl.replay.script_commands`
  does for a script;
- the carried arm's four joints and the claw's finger, which no policy drives
  (`five_foot/claw.py`), as PD position targets, moved towards their goal at a
  fixed rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO / "jumper/tasks/jumper/five_foot/out/example/model_86600.pt"
CAMERA = "robot/onboard"

#: Arm presets: (J0, J1, J2, J3) of the carried left-front arm, radians.
#: `stow` is the task's hold (`claw.py::LF_GRASP`); `reach` and `lift` are the
#: grasp presets of docs/GRASP.md.
ARM_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "stow": (-0.5236, -1.5708, -0.5236, -1.3090),
    "reach": (-1.1013, -1.3142, -0.8291, 0.6949),
    "lift": (-1.1013, -2.0142, -0.8291, 0.6949),
}
#: The policy fine-tuned to walk with the arm at `reach` (docs/GRASP.md), if trained.
CHECKPOINT_REACH = REPO / "oceanking/tasks/ocean/five_foot/out/reach/model.pt"
#: Arm joint speed while moving to a preset, rad/s (rl-wbc-fsm's interpolation).
ARM_RATE = 1.5
#: The finger torque the policy is shown, N*m: what it saw in training, an empty
#: claw (upstream `claw.py`: 0.014 +- 0.056 in the normaliser).
FINGER_TORQUE_SEEN = 0.014


def _hide_finger_torque(cfg) -> None:
    """Show the policy the empty claw's finger torque, whatever the claw squeezes.

    The finger's torque is observed and the policy only ever trained with an empty
    claw; a squeezed piece saturates the servo (1.2 N*m, ~21 standard deviations
    out) and the gait comes apart: upstream measured the turn halved, and fed the
    training value it turned nearly as well as with an empty claw (`claw.py`). In
    the demo the pieces slipped out 1-3 s into the first turn or walk after the
    grasp (3 of 3 scripted, 2026-11-01). The grip keeps its full force."""
    from tasks.jumper.five_foot.claw import FINGER_JOINT

    for group in cfg.observations.values():
        term = group.terms.get("actuator_force")
        if term is None:
            continue
        func = term.func

        def hidden(env, func=func, **params):
            out = func(env, **params).clone()
            asset_cfg = params.get("asset_cfg")
            asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
            finger = asset.find_actuators([FINGER_JOINT])[0][0]
            ids = asset_cfg.actuator_ids if asset_cfg is not None else None
            if ids is None or isinstance(ids, slice):
                col = finger
            else:
                col = list(ids).index(finger)
            out[:, col] = FINGER_TORQUE_SEEN
            return out

        term.func = hidden


@dataclass
class Command:
    twist: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pose: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


class OceanSim:
    def __init__(self, checkpoint: Path | str = CHECKPOINT, *, beta: float = 0.8,
                 current: tuple[float, float] = (0.0, 0.0), scene: str = "seabed",
                 camera_size: tuple[int, int] = (640, 480), presets: dict | None = None,
                 arm: str = "stow", layout: str = "full", hide_finger_torque: bool = True):
        import torch
        from mjrl.backend import resolve
        from mjrl.backend.select import use_backend

        import tasks

        use_backend(resolve(backend="native", device="cpu", num_envs=1))
        from dataclasses import asdict

        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from mjrl.replay import skip_rewards

        import scenes

        cfg = tasks.load_env_cfg("ocean.five_foot", play=True, task_args={
            "buoyancy_ratio": beta, "current": list(current), "demo_scene": layout, "arm": arm})
        if scene:
            scenes.apply(cfg, scene)
        if hide_finger_torque:
            _hide_finger_torque(cfg)
        cfg.scene.num_envs = 1
        cfg.episode_length_s = 1e6  # the demo ends when the agent says so
        self.env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
        skip_rewards(self.env)
        agent_cfg = tasks.load_agent_cfg("ocean.five_foot")
        self.wrapped = RslRlVecEnvWrapper(self.env, clip_actions=agent_cfg.clip_actions)
        runner_cls = tasks.load_runner_cls("ocean.five_foot")
        runner = runner_cls(self.wrapped, asdict(agent_cfg), device="cpu")
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cpu")
        self.policy = runner.get_inference_policy(device="cpu")
        self._torch = torch

        self.cmd = Command()
        self._install_commands()
        from tasks.jumper.five_foot.claw import ARM_JOINTS, FINGER_JOINT, GRIPPER_CLOSED, GRIPPER_OPEN

        self.robot = self.env.scene["robot"]
        self._arm_ids = torch.as_tensor(self.robot.find_joints(list(ARM_JOINTS), preserve_order=True)[0])
        self._finger_id = torch.as_tensor(self.robot.find_joints([FINGER_JOINT])[0])
        self.gripper_open, self.gripper_closed = GRIPPER_OPEN, GRIPPER_CLOSED
        self.presets = {**ARM_PRESETS, **(presets or {})}
        self.arm_mode = arm
        self.arm_goal = np.array(self.presets[arm])
        self.arm_now = np.array(self.presets[arm])
        self.finger_goal = self.finger_now = GRIPPER_OPEN
        self.obs = self.wrapped.get_observations()
        self.time = 0.0
        self._camera_size = camera_size
        self._render = None
        self._video = None
        from oceanking.demo.scene import LAYOUTS, TRASH

        self.layout = layout
        self.zone = LAYOUTS[layout]["zone"]
        self.trash = [t.name for t in TRASH if t.name in LAYOUTS[layout]["trash"]]
        self._rest_z = {t.name: t.half_height for t in TRASH if t.name in self.trash}

    # ── commands ────────────────────────────────────────────────────────────

    def _install_commands(self) -> None:
        torch = self._torch
        cm = self.env.command_manager
        twist = cm.get_term("twist")
        pose = cm.get_term("body_pose")
        for term, attr, key in ((twist, "vel_command_b", "twist"), (pose, "pose_command_b", "pose")):
            compute = term.compute

            def scripted(dt, env_ids=None, compute=compute, term=term, attr=attr, key=key):
                compute(dt, env_ids)
                values = torch.tensor(getattr(self.cmd, key), dtype=torch.float32)
                getattr(term, attr)[:, : len(values)] = values
                if hasattr(term, "pose_target_b") and key == "pose":
                    term.pose_target_b[:] = values
                if hasattr(term, "is_standing_env") and key == "twist":
                    term.is_standing_env[:] = bool(np.linalg.norm(self.cmd.twist) < 1e-6)

            term.compute = scripted

    # ── stepping ────────────────────────────────────────────────────────────

    @property
    def step_dt(self) -> float:
        return float(self.env.step_dt)

    def step(self, seconds: float) -> None:
        torch = self._torch
        n = max(1, int(round(seconds / self.step_dt)))
        for _ in range(n):
            # The carried arm and the claw: towards their goals at ARM_RATE.
            d = ARM_RATE * self.step_dt
            self.arm_now = self.arm_now + np.clip(self.arm_goal - self.arm_now, -d, d)
            self.finger_now += float(np.clip(self.finger_goal - self.finger_now, -2 * d, 2 * d))
            self.robot.set_joint_position_target(
                torch.tensor([self.arm_now], dtype=torch.float32), joint_ids=self._arm_ids)
            self.robot.set_joint_position_target(
                torch.tensor([[self.finger_now]], dtype=torch.float32), joint_ids=self._finger_id)
            with torch.inference_mode():
                actions = self.policy(self.obs)
            self.obs, _, _, _ = self.wrapped.step(actions)
            self.time += self.step_dt
            if self._video is not None and self.time >= self._next_frame:
                self._next_frame += 1.0 / self._video_fps
                self._video_frame()
        self._render_dirty = True

    def walk(self, vx: float, vy: float, wz: float, seconds: float) -> None:
        self.cmd.twist = [float(vx), float(vy), float(wz)]
        self.step(seconds)
        self.cmd.twist = [0.0, 0.0, 0.0]

    def stand(self, seconds: float) -> None:
        self.cmd.twist = [0.0, 0.0, 0.0]
        self.step(seconds)

    def set_pose(self, pitch: float = 0.0, roll: float = 0.0, twist: float = 0.0) -> None:
        self.cmd.pose = [float(pitch), float(roll), float(twist)]

    def arm(self, preset: str, settle: float = 0.3) -> None:
        self.arm_goal = np.array(self.presets[preset])
        travel = float(np.max(np.abs(self.arm_goal - self.arm_now))) / ARM_RATE
        self.stand(travel + settle)

    def claw(self, closed: bool, settle: float = 0.6) -> None:
        self.finger_goal = self.gripper_closed if closed else self.gripper_open
        self.stand(settle)

    # ── seeing ──────────────────────────────────────────────────────────────

    def camera(self) -> np.ndarray:
        """The onboard camera's RGB image, uint8 [H, W, 3]."""
        import mujoco

        sim = self.env.sim
        model = getattr(sim, "render_model", sim.mj_model)
        if self._render is None:
            w, h = self._camera_size
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
            model.vis.global_.offheight = max(model.vis.global_.offheight, h)
            self._render = (mujoco.Renderer(model, height=h, width=w), mujoco.MjData(model))
        renderer, data = self._render
        data.qpos[:] = sim.data.qpos[0].cpu().numpy()[: model.nq]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=CAMERA)
        return renderer.render().copy()

    def self_mask(self) -> np.ndarray:
        """Pixels of the last `camera()` image that are the robot itself: the robot
        knows its own body, and a red crab in its own view is not trash."""
        import mujoco

        sim = self.env.sim
        model = getattr(sim, "render_model", sim.mj_model)
        if getattr(self, "_seg", None) is None:
            w, h = self._camera_size
            self._seg = mujoco.Renderer(model, height=h, width=w)
            self._seg.enable_segmentation_rendering()
            robot = model.body("robot/base_link").id
            root = model.body_rootid
            self._robot_geoms = np.array([g for g in range(model.ngeom)
                                          if root[model.geom_bodyid[g]] == root[robot]])
        _, data = self._render
        self._seg.update_scene(data, camera=CAMERA)
        ids = self._seg.render()[..., 0]
        return np.isin(ids, self._robot_geoms)

    def record(self, path: str, fps: float = 15.0, size=(960, 540), distance: float = 1.6,
               elevation: float = -30.0, azimuth: float = 210.0, inset: bool = True) -> None:
        """Record an external view following the robot to `path` (MP4), with the
        onboard camera inset in the corner, until `stop_recording`."""
        import imageio.v2 as iio
        import mujoco

        sim = self.env.sim
        model = getattr(sim, "render_model", sim.mj_model)
        w, h = size
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
        model.vis.global_.offheight = max(model.vis.global_.offheight, h)
        self._video = iio.get_writer(path, fps=fps, macro_block_size=1)
        self._video_fps = fps
        self._next_frame = self.time
        self._video_r = mujoco.Renderer(model, height=h, width=w)
        self._video_cam = mujoco.MjvCamera()
        self._video_cam.distance, self._video_cam.elevation = distance, elevation
        self._video_cam.azimuth = azimuth
        self._video_inset = inset
        self.caption = ""

    def _video_frame(self) -> None:
        from PIL import Image, ImageDraw

        if self._render is None:
            self.camera()
        _, data = self._render
        model = getattr(self.env.sim, "render_model", self.env.sim.mj_model)
        data.qpos[:] = self.env.sim.data.qpos[0].cpu().numpy()[: model.nq]
        import mujoco

        mujoco.mj_forward(model, data)
        x, y, _ = self.pose()
        self._video_cam.lookat[:] = [x, y, 0.08]
        self._video_r.update_scene(data, self._video_cam)
        frame = Image.fromarray(self._video_r.render())
        if self._video_inset:
            onboard = Image.fromarray(self.camera()).resize((256, 192))
            frame.paste(onboard, (frame.width - 266, 10))
        draw = ImageDraw.Draw(frame)
        draw.text((12, 10), f"t={self.time:5.1f}s  {self.caption}", fill=(255, 255, 255))
        self._video.append_data(np.asarray(frame))

    def stop_recording(self) -> None:
        if self._video is not None:
            self._video.close()
            self._video_r.close()
            self._video = None

    def camera_fov(self) -> tuple[float, float]:
        """(horizontal, vertical) field of view of the rendered image, radians."""
        sim = self.env.sim
        model = getattr(sim, "render_model", sim.mj_model)
        fovy = math.radians(float(model.cam_fovy[model.camera(CAMERA).id]))
        w, h = self._camera_size
        return 2 * math.atan(math.tan(fovy / 2) * w / h), fovy

    def camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World position and rotation matrix (columns: camera x right, y up, -z view)."""
        import mujoco

        self.camera()  # brings the render data up to date
        model = getattr(self.env.sim, "render_model", self.env.sim.mj_model)
        _, data = self._render
        cid = model.camera(CAMERA).id
        del mujoco
        return data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3).copy()

    # ── ground truth (tests, scoring) ───────────────────────────────────────

    def pose(self) -> tuple[float, float, float]:
        d = self.robot.data
        p = (d.root_link_pos_w[0] - self.env.scene.env_origins[0]).cpu().numpy()
        q = d.root_link_quat_w[0].cpu().numpy()
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        return float(p[0]), float(p[1]), yaw

    def positions(self) -> dict[str, np.ndarray]:
        o = self.env.scene.env_origins[0]
        return {n: (self.env.scene[n].data.root_link_pos_w[0] - o).cpu().numpy()
                for n in self.trash}

    #: How far inside the jaw tips an object sits when gripped, m.
    MOUTH_DEPTH = 0.025

    def jaws(self) -> dict[str, np.ndarray]:
        """World positions (env frame) of the claw's hinge and its two jaw tips."""
        from tasks.jumper.five_foot.claw import CARRIED_LEG as L

        names = {"hinge": f"{L}_finger_link", "finger_tip": f"{L}_finger_tip_link",
                 "palm_tip": f"{L}_palm_pad_f_link"}
        o = self.env.scene.env_origins[0].cpu().numpy()
        pos = self.robot.data.body_link_pos_w[0].cpu().numpy()
        return {k: pos[self.robot.find_bodies([n])[0][0]] - o for k, n in names.items()}

    def mouth(self) -> np.ndarray:
        """Where a gripped object sits: `MOUTH_DEPTH` inside the midpoint of the two
        jaw tips, towards the hinge (env frame, m)."""
        j = self.jaws()
        tips = (j["finger_tip"] + j["palm_tip"]) / 2
        inward = j["hinge"] - tips
        inward[2] = 0.0
        return tips + self.MOUTH_DEPTH * inward / max(np.linalg.norm(inward), 1e-9)

    def holding(self) -> str | None:
        """The trash within 4 cm of the mouth (horizontally) and off the floor (the
        arm-out policy carries it only ~6 mm up)."""
        m = self.mouth()
        for name, p in self.positions().items():
            if np.linalg.norm(p[:2] - m[:2]) < 0.04 and p[2] > self._rest_z[name] + 0.003:
                return name
        return None

    def close(self) -> None:
        self.stop_recording()
        if self._render is not None:
            self._render[0].close()
        if getattr(self, "_seg", None) is not None:
            self._seg.close()
        self.env.close()


__all__ = ["ARM_PRESETS", "CHECKPOINT", "OceanSim"]
