"""Upstream `scripts/play.py`, with what the water does drawn into its video.

    cd jumper && MUJOCO_GL=egl ../.venv/bin/python -m oceanking.tools.showcase_video \\
        --current -0.2 0 --command-script CMD.py -- <play.py arguments>

The current cannot be seen and a following camera hides how far a robot got, so
three things are added to every frame, all in the world frame of the recorded
environment:

- **marine snow**: specks suspended in the water, carried by the current (a still
  speck in still water; the robot walks past them);
- **the commanded position**: a green ring where the robot would be if it walked
  exactly as commanded (the command script integrated from the start pose);
- **the path walked** (yellow dots) and a white start line.

They are drawn by the renderer only: the simulation, the policy and the
trajectory written by `play.py` are untouched.
"""

from __future__ import annotations

import argparse
import math
import runpy
import sys
from pathlib import Path

import numpy as np

SNOW = 300  #: specks
BOX = 2.4  #: side of the box of water around the robot that holds them, m
SNOW_TOP = 0.5  #: highest speck, m
SNOW_RGBA = (0.92, 0.95, 1.0, 0.75)
TRAIL_RGBA = (1.0, 0.85, 0.15, 0.9)
GHOST_RGBA = (0.2, 1.0, 0.35, 0.55)


def _install(current: tuple[float, float], script) -> None:
    import mujoco
    from mjrl.viewer import video

    rng = np.random.default_rng(0)
    seeds = rng.uniform(0.0, 1.0, size=(SNOW, 3))
    phases = rng.uniform(0.0, 2 * math.pi, size=(SNOW, 2))
    state: dict = {"trail": [], "ghost": None, "t": None, "yaw0": None, "start": None}
    cur = np.array([current[0], current[1], 0.0])
    eye = np.eye(3).reshape(-1)

    def add(scn, gtype, size, pos, rgba, mat=eye):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                            np.asarray(mat, float), np.asarray(rgba, np.float32))
        g.emission = 0.3
        scn.ngeom += 1

    original = video.ReplayVideo._write_frames

    def write_frames(self, count):
        r = self._renderer
        r.update(self._env.sim.data)
        data, model = r._data, r._model
        free = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
        adr = int(model.jnt_qposadr[free])
        pos = data.qpos[adr:adr + 3].copy()
        w, x, y, z = data.qpos[adr + 3:adr + 7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        t = float(data.time)
        if state["t"] is None:
            state.update(t=t, yaw0=yaw, start=pos.copy(), ghost=pos[:2].copy())
        else:
            # The commanded twist is body-frame; it is integrated along the start
            # heading turned by the commanded yaw rate.
            dt = t - state["t"]
            if dt > 0:
                cmd = (script(t)["twist"] if script else (0.0, 0.0, 0.0))
                state["yaw0"] += float(cmd[2]) * dt
                c, s = math.cos(state["yaw0"]), math.sin(state["yaw0"])
                state["ghost"] += dt * np.array([c * cmd[0] - s * cmd[1], s * cmd[0] + c * cmd[1]])
            state["t"] = t
        if not state["trail"] or np.linalg.norm(pos[:2] - state["trail"][-1]) > 0.04:
            state["trail"].append(pos[:2].copy())

        scn = r._renderer.scene
        # No shadows: the specks' shadows read as dirt on the floor.
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        # Marine snow: a fixed field of specks advected by the current, wrapped into a
        # box that moves with the robot (so there are always specks around it).
        lo = pos - np.array([BOX / 2, BOX / 2, 0.0])
        wob = 0.01 * np.stack([np.sin(0.7 * t + phases[:, 0]), np.sin(0.9 * t + phases[:, 1])], 1)
        p = seeds * np.array([BOX, BOX, SNOW_TOP]) + cur * t
        p[:, :2] += wob
        p[:, :2] = lo[:2] + np.mod(p[:, :2] - lo[:2], BOX)
        p[:, 2] = 0.02 + np.mod(p[:, 2], SNOW_TOP)
        for q in p:
            add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.0045, 0, 0), q, SNOW_RGBA)
        # Start line, across the start heading.
        st = state["start"]
        add(scn, mujoco.mjtGeom.mjGEOM_BOX, (0.012, 0.35, 0.003), (st[0], st[1], 0.012),
            (1, 1, 1, 0.9))
        for q in state["trail"]:
            add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, (0.014, 0.002, 0), (q[0], q[1], 0.012),
                TRAIL_RGBA)
        g = state["ghost"]
        add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, (0.10, 0.003, 0), (g[0], g[1], 0.014), GHOST_RGBA)
        add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, (0.012, 0.15, 0), (g[0], g[1], 0.15), GHOST_RGBA)

        frame = r.render()
        for _ in range(count):
            self._writer.send(frame)
            self.frame_count += 1

    video.ReplayVideo._write_frames = write_frames
    video.ReplayVideo._original_write_frames = original


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--current", type=float, nargs=2, default=(0.0, 0.0),
                    help="the current the specks drift with, m/s (pass play.py the same)")
    ap.add_argument("--command-script", default=None,
                    help="the command script play.py runs (for the commanded position)")
    ap.add_argument("play", nargs=argparse.REMAINDER, help="-- then play.py's arguments")
    args = ap.parse_args()
    play = [a for a in args.play if a != "--"]
    script = None
    if args.command_script:
        script = runpy.run_path(args.command_script)["script"]
        play += ["--command-script", args.command_script]
    sys.path[:0] = [str(Path.cwd() / "scripts"), str(Path.cwd())]
    _install(tuple(args.current), script)
    sys.argv = ["scripts/play.py", *play]
    runpy.run_path("scripts/play.py", run_name="__main__")


if __name__ == "__main__":
    main()
