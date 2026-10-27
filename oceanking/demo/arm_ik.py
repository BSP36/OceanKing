"""Inverse kinematics of the carried arm: move the claw's mouth by a small step.

The robot cannot walk with its arm out (the five-foot policy was trained with it
stowed: 1.5 s of 0.15 m/s commanded walks 0 mm with the arm at `reach`, 0.13 m
with it stowed; 2026-10-27), so the last few centimetres onto a piece of trash
are the arm's. This solves joint angles that move the mouth -- 25 mm inside the
midpoint of the two jaw tips, as `OceanSim.mouth` defines it -- by a requested
base-frame offset, on upstream's plain-MuJoCo model of the arm
(`five_foot/tools/grasp_pose.py::Claw`, the robot standing at STAND_Z), by damped
Gauss-Newton on a numerical Jacobian, inside the joints' soft limits.
"""

from __future__ import annotations

import numpy as np

MOUTH_DEPTH = 0.025


class ArmIK:
    def __init__(self):
        from tasks.jumper.five_foot.claw import ARM_JOINTS, GRIPPER_OPEN
        from tasks.jumper.five_foot.tools.grasp_pose import DEFAULT_MODEL, Claw
        from oceanking.demo.sim import REPO

        self._claw = Claw(str(REPO / "jumper" / DEFAULT_MODEL))
        self._joints = ARM_JOINTS
        self._open = GRIPPER_OPEN
        m = self._claw.m
        self._ids = {n: m.body(n).id for n in ("LF_finger_link", "LF_finger_tip_link",
                                                "LF_palm_pad_f_link")}
        self.lo = np.array([self._claw.soft_limits[j][0] + 0.05 for j in ARM_JOINTS])
        self.hi = np.array([self._claw.soft_limits[j][1] - 0.05 for j in ARM_JOINTS])

    def mouth(self, arm) -> np.ndarray:
        """Mouth position for arm angles `arm`, in the standing model's frame (the
        base is at the origin's xy, facing +x)."""
        self._claw.set_pose(list(arm), self._open)
        x = self._claw.d.xpos
        tips = (x[self._ids["LF_finger_tip_link"]] + x[self._ids["LF_palm_pad_f_link"]]) / 2
        inward = x[self._ids["LF_finger_link"]] - tips
        inward[2] = 0.0
        return tips + MOUTH_DEPTH * inward / max(np.linalg.norm(inward), 1e-9)

    def move(self, arm, delta, iters: int = 30, weights=(1.0, 1.0, 0.5)) -> tuple[np.ndarray, float]:
        """Joint angles moving the mouth by `delta` (base frame, m) from `arm`, and
        the residual (m). The vertical is weighted down: height matters less than
        getting the trash between the jaws."""
        q = np.asarray(arm, float).copy()
        goal = self.mouth(q) + np.asarray(delta, float)
        w = np.asarray(weights)
        for _ in range(iters):
            e = (goal - self.mouth(q)) * w
            if np.linalg.norm(e) < 1e-3:
                break
            J = np.zeros((3, len(q)))
            for k in range(len(q)):
                dq = q.copy()
                dq[k] += 1e-3
                J[:, k] = (self.mouth(dq) - self.mouth(q)) * w / 1e-3
            step = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), e)
            q = np.clip(q + np.clip(step, -0.2, 0.2), self.lo, self.hi)
        return q, float(np.linalg.norm(goal - self.mouth(q)))


__all__ = ["ArmIK"]
