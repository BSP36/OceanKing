"""ocean.five_foot -- `jumper.five_foot` on the seabed: five legs, the claw carried.

Upstream's five-foot task, unchanged in observation, action, reward and command,
with `oceanking.hydro` installed on top, so the official checkpoint runs here as
it is: the baseline for walking with the claw, and grasping, under water
(docs/GRASP.md). `--objects` / `--hold` are upstream's; the props get their own
buoyancy and drag (`oceanking.props`).
"""

from __future__ import annotations

from ...jumper.common.assets import JUMPER_ASSETS
from ...registry import register

register(
    id="ocean.five_foot",
    assets=JUMPER_ASSETS,
    description=(
        "jumper.five_foot underwater with the OceanKing ballast kit: the claw carried, "
        "beta, ground friction and current pinned by flags"
    ),
    tags=("locomotion", "jumper", "ocean", "five_foot", "manipulation"),
)


def cli_args(group):
    """Upstream's `--objects` / `--hold` and the water's pins."""
    import os

    from ...jumper.five_foot import cli_args as upstream

    names = upstream(group)

    def env(name):
        return os.environ.get(name, "").strip() or None

    group.add_argument("--buoyancy", dest="buoyancy_ratio", type=float,
                       default=float(env("OCEAN_BUOYANCY") or 0.8), metavar="BETA",
                       help="beta (default 0.8)")
    current = env("OCEAN_CURRENT")
    group.add_argument("--current", dest="current", type=float, nargs=2,
                       default=[float(v) for v in current.split()] if current else [0.0, 0.0],
                       metavar=("VX", "VY"), help="the current, world frame, m/s")
    group.add_argument("--trajectory-out", dest="trajectory_out",
                       default=env("OCEAN_TRAJECTORY_OUT"), metavar="PATH",
                       help="replays: record the root pose every control step")
    group.add_argument("--demo-scene", dest="demo_scene", action="store_true",
                       help="the demo's sunken trash and drop zone (oceanking.demo.scene)")
    group.add_argument("--arm", dest="arm", choices=("stow", "reach", "reach_lift"),
                       default=env("OCEAN_ARM") or "stow",
                       help="where the carried arm is held: upstream's stow, or out at the "
                       "grasp preset (docs/GRASP.md)")
    return (*names, "buoyancy_ratio", "current", "trajectory_out", "demo_scene", "arm")
