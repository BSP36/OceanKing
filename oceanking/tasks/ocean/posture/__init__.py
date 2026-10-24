"""ocean.posture -- `jumper.posture` on the seabed.

The environment is upstream `jumper.posture`'s, unchanged in observation, action,
reward and command, with `oceanking.hydro` installed on top. A checkpoint trained
for `jumper.posture` therefore runs here as it is, which is what the land-to-water
comparison needs: `--hydro off` is the land baseline in this task.
"""

from __future__ import annotations

from ...jumper.common.assets import JUMPER_ASSETS
from ...registry import register

register(
    id="ocean.posture",
    assets=JUMPER_ASSETS,
    description=(
        "jumper.posture underwater with the OceanKing ballast kit: buoyancy, drag "
        "and a current set by --hydro, --buoyancy, --crown-air and --current"
    ),
    tags=("locomotion", "jumper", "ocean", "posture"),
)


def cli_args(group):
    """The water condition, as task arguments (see `tasks.load_cli_args`).

    Each falls back to an `OCEAN_*` variable, so `.env.local` can pin one
    condition for a whole session. Training, replay and evaluation of one
    experiment must be given the same values.
    """
    import os

    def env(name, default):
        return os.environ.get(name, "").strip() or default

    group.add_argument(
        "--hydro",
        dest="hydro_mode",
        choices=("off", "buoyancy", "drag", "full"),
        default=env("OCEAN_HYDRO", "full"),
        help="which water effects are on; off is land (default: full, or OCEAN_HYDRO)",
    )
    group.add_argument(
        "--buoyancy",
        dest="buoyancy_ratio",
        type=float,
        default=float(env("OCEAN_BUOYANCY", "0.8")),
        metavar="BETA",
        help="buoyancy-to-weight ratio rho*V/M of the whole robot, reached by sizing "
        "the keel; 1 is neutral (default: 0.8, or OCEAN_BUOYANCY)",
    )
    group.add_argument(
        "--buoyancy-model",
        dest="buoyancy_model",
        choices=("archimedes", "uniform"),
        default=env("OCEAN_BUOYANCY_MODEL", "archimedes"),
        help="archimedes: per-link hull volumes plus keel and crown (default); "
        "uniform: the same ratio on every body, no kit (diagnostic)",
    )
    group.add_argument(
        "--crown-air",
        dest="crown_air",
        type=float,
        default=float(env("OCEAN_CROWN_AIR", "0.5")),
        metavar="A",
        help="air fraction of the crown's ballast tank, 0 flooded to 1 blown "
        "(default: 0.5, or OCEAN_CROWN_AIR)",
    )
    group.add_argument(
        "--current",
        dest="current",
        type=float,
        nargs=2,
        default=[float(v) for v in env("OCEAN_CURRENT", "0 0").split()],
        metavar=("VX", "VY"),
        help="uniform horizontal water velocity in the world frame, m/s "
        "(default: 0 0, or OCEAN_CURRENT)",
    )
    group.add_argument(
        "--drag-scale",
        dest="drag_scale",
        type=float,
        default=float(env("OCEAN_DRAG_SCALE", "1.0")),
        help="multiplier on the fluid density seen by the drag model (default: 1)",
    )
    group.add_argument(
        "--nominal",
        dest="nominal",
        action="store_true",
        default=env("OCEAN_NOMINAL", "") in ("1", "true", "on", "yes"),
        help="replay the nominal robot from a fixed start, for side-by-side "
        "comparisons: origin, facing +x, no startup randomisation (foot friction at "
        "the asset's 1.2, no COM offset, no encoder bias)",
    )
    group.add_argument(
        "--trajectory-out",
        dest="trajectory_out",
        default=env("OCEAN_TRAJECTORY_OUT", None),
        metavar="PATH",
        help="replay only: record the root pose and command of every control step "
        "to PATH.npz / PATH.json (see oceanking/trajectory.py)",
    )
    return (
        "hydro_mode", "buoyancy_ratio", "buoyancy_model", "crown_air", "current", "drag_scale",
        "nominal", "trajectory_out",
    )
