"""The robot's skills for the demo: what the agent can ask OceanKing to do.

Each skill uses only what the robot can sense about itself and see: the onboard
camera (through `perception.detect`), its own pose by odometry (in simulation the
base's pose, which an odometer on the robot approximates), its joint encoders
(the claw closes less far on something it holds). None reads where the trash is.

- `look()`: an onboard image and what the detector finds in it, with ground
  distance and bearing;
- `turn(degrees)`, `walk(metres, direction)`: odometry-closed moves;
- `approach(label)`: walk to a detected object until the claw's mouth is on it --
  the tracking part: re-detect while it is in view, dead-reckon the last ground
  fix once it is too close to see, put the arm out short of it, creep the rest;
- `grasp()`: close, lift, and report whether something is held (finger angle);
- `release()`: open, stow;
- `go_to(label)`: walk until the mouth is over a detected place (the drop zone).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from oceanking.demo import scene as _scene  # noqa: F401  (registers the target heights)
from oceanking.demo.perception import Detection, detect

#: Walking speeds the five-foot policy tracks under water, m/s and rad/s. It has a
#: wide standing band: measured (native, beta 0.8, 2026-10-27) 0.08 m/s commanded
#: walks 0.002 m/s and 0.10 sideways 0.01; 0.15 walks 0.12 either way; turns below
#: 0.3 rad/s do not turn, 0.4 turns 0.40. So moves are at full speed, and fine
#: positioning is by pulses: 0.15 m/s for 0.2 s moves 23 mm, for 0.3 s 36 mm.
MAX_SPEED = 0.15
MIN_SPEED = 0.15
MAX_TURN = 0.6
MIN_TURN = 0.45
#: Control period of the closed-loop moves, s.
PERIOD = 0.25
#: Re-detect the target while it is farther than this from the mouth, m.
TRACK_RANGE = 0.45
#: Arm-out stand-off: stop with the target this far *beyond* the mouth's reach
#: position before putting the arm out, m. The stowed claw sticks out ~0.22 m ahead
#: of the base, so a stand-off short of the mouth walked it into the trash and
#: knocked it over (2026-10-27); 0.09 m beyond keeps the target at ~0.35 m.
STANDOFF = 0.09
#: Mouth-on-target tolerance, m (the jaws are 75 mm apart at the tips, the trash
#: 48-52 mm across).
TOLERANCE = 0.012
#: How far inside the mouth the trash is brought, m.
GRIP_DEPTH = 0.015
#: Shoulder lift with a piece in the claw, rad (see `grasp`).
LIFT_RAD = 0.45
#: The finger closes to `GRIPPER_CLOSED` (0.10 rad) on nothing; on a 50 mm object
#: it stops well short (measured: about -0.3 rad).
HOLD_MARGIN = 0.15


@dataclass
class Log:
    lines: list[str] = field(default_factory=list)

    def __call__(self, msg: str) -> None:
        self.lines.append(msg)


#: Pulse length for a distance, s, measured for the arm-out policy at 0.15 m/s from
#: standing (2026-10-27): 0.2 s -> 11 mm, 0.3 s -> 20 mm, 0.5 s -> 55 mm.
_PULSE_D = (0.0, 0.011, 0.020, 0.055)
_PULSE_T = (0.12, 0.20, 0.30, 0.50)


def _pulse(distance: float) -> float:
    return float(np.interp(distance, _PULSE_D, _PULSE_T))


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class Skills:
    def __init__(self, sim, verbose: bool = False):
        self.sim = sim
        self.log = Log()
        self.verbose = verbose
        self.targets: dict[str, tuple[float, float]] = {}  # last ground fix per label
        self.held: str | None = None  # what the last successful grasp picked up
        self._approached: str | None = None
        self._whole: dict[str, bool] = {}
        self.mouth_offset = self._calibrate_mouth()

    def _say(self, msg: str) -> None:
        self.log(msg)
        if self.verbose:
            print(f"[skills t={self.sim.time:6.1f}] {msg}")

    @property
    def carry_pose(self) -> str:
        """Where the arm rides while walking: the pose the policy walks with."""
        return self.sim.arm_mode

    def _calibrate_mouth(self) -> np.ndarray:
        """The mouth in the base frame at `reach`: the robot's own arm kinematics,
        read once by putting the arm out."""
        self.sim.arm("reach", settle=0.4)
        offset = self.to_base(self.sim.mouth()[:2])
        self.sim.arm(self.carry_pose, settle=0.3)
        return offset

    # ── the claw, as its encoder tells it ─────────────────────────────────

    def claw_closed_on_something(self) -> bool:
        finger = float(self.sim.robot.data.joint_pos[0, self.sim._finger_id])
        return self.sim.finger_goal > 0 and finger < self.sim.gripper_closed - HOLD_MARGIN

    def _check_drop(self, msg: str) -> str:
        """Append a warning if what the claw held is no longer in it."""
        if self.held and not self.claw_closed_on_something():
            dropped = self.held
            self.held = None
            self.targets.pop(dropped, None)
            self._say(f"dropped the {dropped}")
            return (msg + f". WARNING: the claw has closed on nothing -- the {dropped} was "
                    "dropped on the way; look for it again")
        return msg

    # ── frames ──────────────────────────────────────────────────────────────

    def to_base(self, xy) -> np.ndarray:
        x, y, yaw = self.sim.pose()
        d = np.asarray(xy, float) - np.array([x, y])
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])

    # ── seeing ──────────────────────────────────────────────────────────────

    def detections(self) -> list[Detection]:
        sim = self.sim
        img = sim.camera()
        pos, mat = sim.camera_pose()
        _, fovy = sim.camera_fov()
        dets = detect(img, fovy, pos, mat, sim.pose(), ignore=sim.self_mask())
        # Per label, the largest blob's ground fix, a whole one first; one partly
        # hidden at its bottom is kept only when there is no better fix yet.
        seen = set()
        for d in sorted(dets, key=lambda d: (not d.whole, -d.pixels)):
            if d.ground is None or d.label in seen or d.label == self.held:
                continue  # (what the claw holds is in view, and is not on the floor)
            if d.whole or d.label not in self.targets or not self._whole.get(d.label, False):
                self.targets[d.label] = d.ground
                self._whole[d.label] = d.whole
            seen.add(d.label)
        self._last_image = img
        return dets

    def look(self) -> tuple[np.ndarray, list[Detection]]:
        dets = self.detections()
        return self._last_image, dets

    # ── moves ───────────────────────────────────────────────────────────────

    def turn(self, degrees: float) -> str:
        """Turn in place, continuously: pulses do not turn (the gait needs ~0.5 s
        to start), so the command holds until the heading is within the coast."""
        start = self.sim.pose()[2]
        goal = start + math.radians(degrees)
        for _ in range(int(60 / PERIOD)):
            err = _wrap(goal - self.sim.pose()[2])
            if abs(err) < math.radians(6):
                break
            # Slowly with a piece in the claw: a full-speed turn swung one out (run 1
            # with Claude, 2026-10-28).
            fast = abs(err) > math.radians(25) and not self.held
            wz = math.copysign(MAX_TURN if fast else MIN_TURN, err)
            self.sim.walk(0.0, 0.0, wz, PERIOD / 2)
        self.sim.stand(0.4)
        turned = math.degrees(_wrap(self.sim.pose()[2] - start))
        self._say(f"turned {turned:+.0f} deg")
        return self._check_drop(f"turned {turned:+.0f} deg")

    def walk(self, metres: float, direction: str = "forward") -> str:
        dirs = {"forward": (1, 0), "back": (-1, 0), "left": (0, 1), "right": (0, -1)}
        ux, uy = dirs[direction]
        x0, y0, _ = self.sim.pose()
        for _ in range(int(120 / PERIOD)):
            x, y, _ = self.sim.pose()
            done = math.hypot(x - x0, y - y0)
            if done >= metres - 0.01:
                break
            self.sim.walk(ux * MAX_SPEED, uy * MAX_SPEED, 0.0,
                          min(PERIOD, max(0.12, (metres - done) / 0.12)))
        self.sim.stand(0.3)
        x, y, _ = self.sim.pose()
        msg = f"walked {math.hypot(x - x0, y - y0):.2f} m {direction}"
        self._say(msg)
        return self._check_drop(msg)

    def _servo(self, label: str, offset: np.ndarray, tol: float, timeout: float,
               track: bool, track_near: bool = False) -> float:
        """Walk until the base-frame point `offset` is on the target's ground fix.
        Far: full speed, facing the target so the camera keeps it, re-detecting.
        Near: pulses at the walking speed, each as long as the error asks, then a
        settle. Returns the remaining error, m."""
        t_end = self.sim.time + timeout
        err = np.inf
        while self.sim.time < t_end:
            if label not in self.targets:
                raise RuntimeError(f"no fix on {label}")
            t = self.to_base(self.targets[label])
            far = np.linalg.norm(t) > TRACK_RANGE + np.linalg.norm(offset)
            if track and (far or track_near):
                self.detections()
                t = self.to_base(self.targets[label])
            e = t - offset
            err = float(np.linalg.norm(e))
            if err < tol:
                break
            heading = math.atan2(t[1], t[0])
            if far:
                # Face it first (turning in place, continuously: a turn needs ~0.5 s
                # to start), then walk forward with a sideways trim. A diagonal or
                # backward command while turning barely moves the arm-out gait with
                # a load (200 s for 0 m towards the drop zone, 2026-10-27).
                if abs(heading) > math.radians(25):
                    self.sim.walk(0.0, 0.0, math.copysign(MAX_TURN, heading), PERIOD)
                    continue
                wz = math.copysign(MIN_TURN, heading) if abs(heading) > math.radians(10) else 0.0
                vy = float(np.clip(2.5 * e[1], -0.07, 0.07))
                self.sim.walk(MAX_SPEED, vy, wz, PERIOD)
            else:
                # One axis at a time at exactly the walking speed: a diagonal or a
                # slow command sits in the standing band and does not move at all
                # (an approach stalled for 17 s on (0.13, -0.08), 2026-10-27). A
                # pulse includes ~0.15 s for the gait to start.
                # Line up sideways only while still well short: an open jaw moving
                # sideways next to the trash knocks it over. Once close, go
                # straight in; if it has drifted sideways there, back off.
                if abs(e[1]) > tol and e[0] > 0.05:
                    k, d = 1, e[1]
                elif e[0] > tol:
                    k, d = 0, e[0]
                elif abs(e[1]) > tol:
                    k, d = 0, -0.06
                else:
                    k, d = 0, e[0]
                v = [0.0, 0.0]
                v[k] = math.copysign(MIN_SPEED, d)
                self.sim.walk(v[0], v[1], 0.0, float(np.clip(abs(d) / 0.11 + 0.15, 0.2, 0.6)))
                self.sim.stand(0.3)
        self.sim.stand(0.3)
        return err

    def approach(self, label: str) -> str:
        """Bring the open claw onto `label`'s ground fix (from the last `look`).

        With the arm-out policy the claw is open in front while walking, so the
        trash enters the mouth from the front. With the official (stowed) policy
        the robot stops with the trash beyond the stowed claw and puts the arm out
        there -- which sweeps the trash aside (docs/GRASP.md): kept as the fallback."""
        if label not in self.targets:
            self.detections()
        if label not in self.targets:
            return f"cannot approach {label}: not in view. Look around first."
        self.sim.claw(closed=False, settle=0.2)
        self._approached = label
        if self.carry_pose == "reach":
            err = self._approach_open(label)
        else:
            self.sim.arm("stow", settle=0.2)
            short = self.mouth_offset + np.array([STANDOFF, 0.0])
            self._servo(label, short, 0.03, timeout=60.0, track=True)
            self.sim.arm("reach", settle=0.4)
            err = self._servo(label, self.mouth_offset, TOLERANCE, timeout=20.0, track=False)
        msg = f"claw at {label}, {err * 100:.1f} cm off its last seen position"
        self._say(msg)
        return msg

    def _approach_open(self, label: str, timeout: float = 90.0) -> float:
        """The arm-out policy's approach: walk the open claw in along its jaws.

        Far, `_servo` (full speed, facing it, re-detecting). Then a continuous walk
        forward with a sideways component that lines the mouth up on the way in
        -- sideways pulses from standing do not move this gait at all (0 mm from
        0.4 s pulses, 2026-10-27), while a sideways component on a moving gait
        does -- stopping 4 cm short; forward pulses for the rest; and the last
        sideways centimetres by the arm (`arm_ik`), standing."""
        from oceanking.demo.arm_ik import ArmIK

        t_end = self.sim.time + timeout
        # Aim the trash 1.5 cm deeper than the mouth: the jaws hold best inside
        # their tips, and the fix runs ~1.4 cm short at the end (a can 2.2 cm
        # ahead of the mouth slipped out of the closing tips, 2026-10-27).
        aim = self.mouth_offset - np.array([GRIP_DEPTH, 0.0])
        while self.sim.time < t_end:
            t = self.to_base(self.targets[label])
            if np.linalg.norm(t) > 0.45:  # far enough to see it whole: refresh the fix
                self.detections()
                t = self.to_base(self.targets[label])
            e = t - aim
            if e[0] <= 0.06:
                break
            # Face so that the target lies on the *mouth's* line, not the body's: the
            # mouth is ~7.6 cm left of the centre line, and facing the target with
            # the body left the mouth that far off for targets on the right -- past
            # what the arm trims, and a jaw knocked them over (2026-10-27).
            r = float(np.linalg.norm(t))
            heading = math.atan2(t[1], t[0]) - math.asin(
                float(np.clip(self.mouth_offset[1] / max(r, 1e-6), -1.0, 1.0)))
            if r > 0.5 and abs(heading) > math.radians(25):
                self.sim.walk(0.0, 0.0, math.copysign(MAX_TURN, heading), PERIOD)
                continue
            wz = 0.0
            if r > 0.5 and abs(heading) > math.radians(6):
                wz = math.copysign(MIN_TURN, heading)
            vy = float(np.clip(2.5 * e[1], -0.08, 0.08))
            self.sim.walk(MAX_SPEED, vy, wz, PERIOD)
        self.sim.stand(0.4)
        for _ in range(8):
            e = self.to_base(self.targets[label]) - aim
            if -0.012 <= e[0] <= 0.012 or self.sim.time > t_end:
                break
            self.sim.walk(math.copysign(MIN_SPEED, e[0]), 0.0, 0.0, _pulse(abs(e[0])))
            self.sim.stand(0.3)
        e = self.to_base(self.targets[label]) - aim
        if 0.006 < abs(e[1]) <= 0.035:
            if not hasattr(self, "_ik"):
                self._ik = ArmIK()
            q, _ = self._ik.move(self.sim.presets["reach"], (0.0, float(e[1]), 0.0),
                                 weights=(1.0, 1.0, 0.3))
            self.sim.presets["aim"] = tuple(float(v) for v in q)
            self.sim.arm("aim", settle=0.4)
            e = e - np.array([0.0, e[1]])
        return float(np.linalg.norm(e))

    def grasp(self) -> str:
        self.sim.claw(closed=True, settle=1.0)
        # Lift from wherever the arm is aimed, by what the arm-out policy was
        # trained to walk with: the shoulder up to 0.5 rad (`--arm reach_lift`),
        # 0.45 here, the trash ~3 cm off the floor. (A 0.7 rad lift left the first
        # arm-out policy unable to walk; within its +-0.1 rad the trash, 6-9 mm up,
        # dragged and anchored it.) The stowed policy does not walk with the arm
        # out anyway.
        raise_by = LIFT_RAD if self.carry_pose == "reach" else 0.7
        lift = np.array(self.sim.arm_goal) + np.array([0.0, -raise_by, 0.0, 0.0])
        self.sim.presets["lift_now"] = tuple(float(v) for v in lift)
        self.sim.arm("lift_now", settle=0.6)
        finger = float(self.sim.robot.data.joint_pos[0, self.sim._finger_id])
        held = finger < self.sim.gripper_closed - HOLD_MARGIN
        self.held = self._approached if held else None
        msg = (f"holding something (claw stopped at {finger:+.2f} rad)" if held else
               f"nothing in the claw (closed to {finger:+.2f} rad)")
        self._say(msg)
        return msg

    def release(self) -> str:
        self.sim.arm("reach", settle=0.3)
        self.sim.claw(closed=False, settle=1.0)
        self.sim.arm(self.carry_pose, settle=0.3)
        if self.held:
            self.targets.pop(self.held, None)  # it is somewhere new now
        self.held = None
        self._say("released")
        return "claw opened and arm stowed"

    def back_off(self, metres: float = 0.15) -> str:
        return self.walk(metres, "back")

    def go_to(self, label: str) -> str:
        """Bring the claw's mouth over `label` (the drop zone), holding what it holds.
        Seen without a distance (its foot hidden), walk 0.3 m towards its bearing
        and look again, up to five times."""
        for _ in range(5):
            if label in self.targets:
                break
            dets = [d for d in self.detections() if d.label == label]
            if label in self.targets:
                break
            if not dets or dets[0].bearing is None:
                return f"cannot go to {label}: not in view. Look around first."
            if abs(dets[0].bearing) > 10:
                self.turn(dets[0].bearing)
            self.walk(0.3, "forward")
        if label not in self.targets:
            return f"cannot go to {label}: it stays without a distance; get closer."
        # The zone is large and flat: keep refreshing its fix all the way in (the
        # far-only refresh left pieces 0.3-0.4 m short of it, 2026-10-27).
        err = self._servo(label, self.mouth_offset, 0.04, timeout=90.0, track=True,
                          track_near=True)
        msg = f"claw over {label}, {err * 100:.0f} cm off its last seen position"
        self._say(msg)
        if self.held:
            msg = self._check_drop(msg + f", holding the {self.held}")
        return msg

    def status(self) -> str:
        x, y, yaw = self.sim.pose()
        finger = float(self.sim.robot.data.joint_pos[0, self.sim._finger_id])
        if self.held and self.claw_closed_on_something():
            held = f"; holding the {self.held}"
        elif self.sim.finger_goal > 0:
            held = "; claw closed on nothing"
        else:
            held = "; claw open and empty"
        return (f"time {self.sim.time:.1f} s{held}; odometry x {x:+.2f} m, y {y:+.2f} m, heading "
                f"{math.degrees(yaw):+.0f} deg; claw at {finger:+.2f} rad "
                f"({'closed' if self.sim.finger_goal > 0 else 'open'}); known fixes: "
                + ", ".join(f"{k} at ({v[0]:+.2f}, {v[1]:+.2f})" for k, v in self.targets.items()))


__all__ = ["Skills"]
