"""Record what a replay did: the robot's root pose and command every control step.

Installed by a task as an mjlab event term in ``step`` mode, so it runs inside
upstream's `scripts/play.py` without that script knowing about it. Written once,
when the process exits, as ``<path>.npz`` (arrays) and ``<path>.json`` (metadata).

Positions are relative to each environment's origin. Frames: world, right-handed,
z up, metres; quaternions are MuJoCo's (w, x, y, z). The pose is the one the step
event sees, which lags the last physics substep by one substep (mjlab's
documented staleness), uniformly for every step.
"""

from __future__ import annotations

import atexit
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "oceanking-trajectory/1"


class TrajectoryRecorder:
    def __init__(self, cfg: Any, env: Any) -> None:
        # The given path is a stem: ".npz" / ".json" are appended, never substituted
        # (`with_suffix` turned "b0.75_mu0.4" into "b0.75_mu0.npz", so runs that
        # differed after the last dot overwrote each other).
        self._path = Path(cfg.params["path"])
        self._meta = dict(cfg.params.get("meta", {}))
        self._step_dt = float(env.step_dt)
        self._robot = env.scene["robot"]
        self._origins = env.scene.env_origins
        self._t: list[float] = []
        self._pos: list[np.ndarray] = []
        self._quat: list[np.ndarray] = []
        self._cmd: list[np.ndarray] = []
        self._episode_step: list[np.ndarray] = []
        self._count = 0
        atexit.register(self.save)

    def __call__(self, env: Any, env_ids: Any, path: str = "", meta: Any = None) -> None:
        data = self._robot.data
        self._t.append(self._count * self._step_dt)
        self._pos.append((data.root_link_pos_w - self._origins).detach().cpu().numpy())
        self._quat.append(data.root_link_quat_w.detach().cpu().numpy())
        # A task without a velocity command (ocean.jump) records zeros.
        twist = (env.command_manager.get_command("twist")
                 if "twist" in env.command_manager.active_terms else None)
        self._cmd.append(
            np.zeros((self._pos[-1].shape[0], 3)) if twist is None
            else twist[:, :3].detach().cpu().numpy()
        )
        self._episode_step.append(env.episode_length_buf.detach().cpu().numpy().copy())
        self._count += 1

    def save(self) -> None:
        if not self._t:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        npz = self._path.parent / (self._path.name + ".npz")
        np.savez_compressed(
            npz,
            t=np.asarray(self._t),
            root_pos=np.stack(self._pos, axis=1),
            root_quat_wxyz=np.stack(self._quat, axis=1),
            command_twist=np.stack(self._cmd, axis=1),
            episode_step=np.stack(self._episode_step, axis=1),
        )
        meta = {
            "schema": SCHEMA,
            "step_dt_s": self._step_dt,
            "steps": len(self._t),
            "frames": "world, z up, metres, relative to each env origin; quat wxyz",
            "arrays": {
                "t": "[T] s",
                "root_pos": "[N, T, 3] m",
                "root_quat_wxyz": "[N, T, 4]",
                "command_twist": "[N, T, 3] body-frame vx, vy (m/s), wz (rad/s)",
                "episode_step": "[N, T] control steps since the env's last reset",
            },
            **self._meta,
        }
        (self._path.parent / (self._path.name + ".json")).write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[oceanking] trajectory {npz}")
