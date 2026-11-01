"""Finding the trash and the drop zone in the onboard camera's image.

A deliberately plain detector: each target has a colour no other thing on the
seabed has (`scene.py`), so a hue/saturation threshold, connected components and
a ground-plane projection find it -- no learned model to download or to be wrong
in ways nobody sees. What it needs from the robot is what the robot knows about
itself: the camera's intrinsics (field of view) and its pose (from the body's
kinematics). It never reads where the trash is.

The one confusable thing is the reference scene's stakes: tall, thin and
red-brown. A blob more than `MAX_ASPECT` times as tall as it is wide is a pole,
not trash.

Distance comes from the blob's lowest pixel: that is where the object meets the
ground, and the ray through it meets the plane z = 0 there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: name -> (hue range in degrees, minimum saturation, minimum value). Hue wraps.
#: Calibrated on the onboard camera at 1 m in the seabed scene, where the water's
#: light leaves everything dark (2026-10-27, from segmentation masks): can hue
#: 356-359 sat 0.64-0.87 value 0.21-0.29; bottle 141-159, 0.60-0.90, 0.20-0.33;
#: carton 72, 0.93, 0.22.
CLASSES: dict[str, tuple[tuple[float, float], float, float]] = {
    "red_bottle": ((-15.0, 8.0), 0.55, 0.10),     # red
    "green_bottle": ((115.0, 170.0), 0.50, 0.10),  # green
    "carton": ((55.0, 90.0), 0.75, 0.10),     # yellow
    "drop_zone": ((262.0, 330.0), 0.50, 0.10),  # magenta: 268-283 in the water light
}
MIN_PIXELS = 12
#: A red blob this many times taller than wide is one of the scene's stakes, not
#: the red bottle (only red is shared with them).
MAX_ASPECT = 5.0


@dataclass
class Detection:
    label: str
    pixels: int
    box: tuple[int, int, int, int]  # u0, v0, u1, v1
    distance: float | None  # m, along the ground from the camera
    bearing: float | None  # deg, positive to the left of the robot's heading
    ground: tuple[float, float] | None  # world xy of the contact point
    #: False when the blob touches the image border or the robot's own body: its
    #: lowest pixel may not be where it meets the ground.
    whole: bool = True

    def describe(self) -> str:
        if self.distance is None:
            where = "" if self.bearing is None else (
                f", {abs(self.bearing):.0f} deg to the {'left' if self.bearing >= 0 else 'right'}")
            return f"{self.label}: seen{where}, distance unknown (its foot is not in view)"
        side = "left" if self.bearing >= 0 else "right"
        note = "" if self.whole else " (partly hidden: distance approximate)"
        return (f"{self.label}: {self.distance:.2f} m away, {abs(self.bearing):.0f} deg to the "
                f"{side}{note}")


def _hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn + 1e-9
    h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4))
    return h * 60.0, np.where(mx > 0, d / (mx + 1e-9), 0.0), mx


def _mask(h, s, v, hue, smin, vmin) -> np.ndarray:
    lo, hi = hue
    if lo < 0:
        inside = (h >= lo + 360.0) | (h <= hi)
    else:
        inside = (h >= lo) & (h <= hi)
    return inside & (s >= smin) & (v >= vmin)


#: Known heights of the targets, m. (A fix from the top edge at this height was
#: tried for trash whose bottom the claw hides, and dropped: the onboard camera sits
#: ~0.13 m up, level with a can's top, so that ray grazes the plane and a pixel is
#: tens of centimetres -- 0.75 m read for 0.90, 2026-10-27.)
HEIGHTS: dict[str, float] = {}

#: label -> radius, m, of a flat disk lying on the ground (the drop zone; filled in by
#: `scene.py`). The lowest pixel of a disk is its near rim, not its centre: the
#: centre is a radius further along the line of sight. Without this the zone read up
#: to 0.18 m short and pieces were released on its rim (2026-11-01).
FLAT_RADIUS: dict[str, float] = {}


def ground_point(u: float, v: float, size: tuple[int, int], fovy: float,
                 cam_pos: np.ndarray, cam_mat: np.ndarray, z: float = 0.0) -> np.ndarray | None:
    """Where the ray through pixel (u, v) meets the plane at height `z`, world
    frame; None if it does not (above the horizon)."""
    w, h = size
    f = (h / 2) / math.tan(fovy / 2)
    d_cam = np.array([(u - w / 2) / f, -(v - h / 2) / f, -1.0])
    d = cam_mat @ d_cam
    if d[2] >= -1e-6 or cam_pos[2] <= z:
        return None
    t = (z - cam_pos[2]) / d[2]
    return cam_pos + t * d


def detect(rgb: np.ndarray, fovy: float | None = None, cam_pos=None, cam_mat=None,
           robot_pose: tuple[float, float, float] | None = None,
           exclude_rows_below: float = 1.0, ignore: np.ndarray | None = None) -> list[Detection]:
    """Every target class's blobs, largest first. With the camera's geometry and the
    robot's pose, each gets a ground distance and a bearing from the robot.
    `ignore` masks pixels out (the robot's own body, which it knows)."""
    from scipy import ndimage

    h, s, v = _hsv(rgb)
    H, W = rgb.shape[:2]
    out: list[Detection] = []
    for label, (hue, smin, vmin) in CLASSES.items():
        m = _mask(h, s, v, hue, smin, vmin)
        m[int(exclude_rows_below * H):] = False
        if ignore is not None:
            m &= ~ignore
        lab, n = ndimage.label(m)
        for k, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            count = int((lab[sl] == k).sum())
            if count < MIN_PIXELS:
                continue
            v0, v1 = sl[0].start, sl[0].stop
            u0, u1 = sl[1].start, sl[1].stop
            if label == "red_bottle" and (v1 - v0) > MAX_ASPECT * (u1 - u0):
                continue  # a pole, not trash
            border = not (v1 < H - 2 and u0 > 1 and u1 < W - 1)
            hidden = False
            if ignore is not None and not border:
                # Hidden at the bottom: the rows just under the blob are mostly the
                # robot's own body (a blob merely beside it is still whole).
                below = ignore[v1:min(v1 + 8, H), u0:u1]
                hidden = bool(below.size and below.mean() > 0.15)
            whole = not border and not hidden
            dist = bearing = ground = None
            if fovy is not None and cam_pos is not None and robot_pose is not None:
                g = ground_point((u0 + u1) / 2, v1 - 0.5, (W, H), fovy, cam_pos, cam_mat)
                if g is None:
                    # Above the horizon (a pole's top, with its foot hidden): no
                    # distance, but the bearing of the ray still says where it is.
                    w_, h_ = W, H
                    f = (h_ / 2) / math.tan(fovy / 2)
                    ray = cam_mat @ np.array([((u0 + u1) / 2 - w_ / 2) / f, 0.0, -1.0])
                    rel = math.atan2(ray[1], ray[0]) - robot_pose[2]
                    bearing = math.degrees(math.atan2(math.sin(rel), math.cos(rel)))
                if g is not None and label in FLAT_RADIUS:
                    sight = np.asarray(g[:2], float) - np.asarray(cam_pos[:2], float)
                    g = np.array(g, float)
                    g[:2] += FLAT_RADIUS[label] * sight / max(np.linalg.norm(sight), 1e-9)
                if g is not None:
                    x, y, yaw = robot_pose
                    dx, dy = g[0] - x, g[1] - y
                    dist = math.hypot(dx, dy)
                    rel = math.atan2(dy, dx) - yaw
                    bearing = math.degrees(math.atan2(math.sin(rel), math.cos(rel)))
                    ground = (float(g[0]), float(g[1]))
            out.append(Detection(label, count, (u0, v0, u1, v1), dist, bearing, ground, whole))
    out.sort(key=lambda d: -d.pixels)
    return out


__all__ = ["CLASSES", "FLAT_RADIUS", "Detection", "detect", "ground_point"]
