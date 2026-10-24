"""Tile replays of the same command script into one labelled comparison video.

Each panel is a `play.py --video` recording with a matching `--trajectory-out`
recording; the caption shows the condition and, from the trajectory, how far the
robot has moved from its start at that instant. Panels must share fps and size.

    .venv/bin/python -m oceanking.tools.compare_video --out cmp.mp4 --cols 2 \\
        "Land=land.mp4:land.npz" "Water beta 0.8=b08.mp4:b08.npz"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def _font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _panel(spec: str) -> tuple[str, Path, Path]:
    label, _, files = spec.rpartition("=")
    video, _, traj = files.partition(":")
    return label, Path(video), Path(traj)


def summary(traj: Path, windows: dict[str, tuple[float, float]]) -> dict:
    z = np.load(traj)
    t, p = z["t"], z["root_pos"][0]
    out = {"final_xy_m": p[-1, :2].round(3).tolist(),
           "path_length_m": float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum())}
    for name, (a, b) in windows.items():
        i, j = np.searchsorted(t, a), np.searchsorted(t, b)
        v = (p[j, :2] - p[i, :2]) / (t[j] - t[i])
        out[f"mean_velocity_xy_{name}_m_s"] = v.round(3).tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("panels", nargs="+", help="LABEL=VIDEO.mp4:TRAJECTORY.npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--scale", type=float, default=0.5, help="panel scale factor")
    ap.add_argument("--title", default="")
    ap.add_argument("--footer", default="Simulated in MuJoCo, real time. Grid: 0.5 m; "
                    "stakes every 0.5 m, red every 1 m.")
    ap.add_argument("--window", action="append", default=[], metavar="NAME:T0:T1",
                    help="report mean velocity over [T0, T1] s in the summary")
    args = ap.parse_args()

    panels = [_panel(s) for s in args.panels]
    readers = [imageio.get_reader(v) for _, v, _ in panels]
    fps = readers[0].get_meta_data()["fps"]
    trajs = [np.load(t) for _, _, t in panels]
    rows = -(-len(panels) // args.cols)
    first = readers[0].get_data(0)
    h, w = int(first.shape[0] * args.scale), int(first.shape[1] * args.scale)
    bar = 44 if args.title else 0
    foot = 30
    big, mono, small = _font(FONT, 26), _font(FONT_MONO, 22), _font(FONT_MONO, 18)
    writer = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1)
    frame = 0
    while True:
        try:
            images = [r.get_data(frame) for r in readers]
        except IndexError:
            break
        canvas = Image.new("RGB", (w * args.cols, h * rows + bar + foot), (12, 24, 36))
        draw = ImageDraw.Draw(canvas)
        if args.title:
            draw.text((14, 8), args.title, font=big, fill=(240, 240, 240))
        t_now = frame / fps
        for k, (img, (label, _, _), z) in enumerate(zip(images, panels, trajs)):
            x0, y0 = (k % args.cols) * w, bar + (k // args.cols) * h
            canvas.paste(Image.fromarray(img).resize((w, h)), (x0, y0))
            i = min(np.searchsorted(z["t"], t_now), len(z["t"]) - 1)
            p = z["root_pos"][0, i] - z["root_pos"][0, 0]
            draw.rectangle([x0, y0, x0 + w, y0 + 36], fill=(0, 0, 0))
            draw.text((x0 + 10, y0 + 5), label, font=big, fill=(255, 220, 90))
            text = f"t={z['t'][i]:5.1f}s  dx={p[0]:+5.2f}m  dy={p[1]:+5.2f}m"
            draw.rectangle([x0, y0 + h - 32, x0 + w, y0 + h], fill=(0, 0, 0))
            draw.text((x0 + 10, y0 + h - 29), text, font=mono, fill=(230, 230, 230))
        draw.text((10, h * rows + bar + 5), args.footer, font=small, fill=(200, 200, 200))
        writer.append_data(np.asarray(canvas))
        frame += 1
    writer.close()

    windows = {n: (float(a), float(b)) for n, a, b in (s.split(":") for s in args.window)}
    report = {label: summary(t, windows) for label, _, t in panels}
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"{args.out} ({frame} frames at {fps:g} fps)")


if __name__ == "__main__":
    main()
