"""ocean.walk -- jumper.posture, adapted to walk at its command under water.

Training task of `docs/TRAINING.md`: `jumper.posture`'s environment with the water
of `oceanking.hydro`, beta, current and ground friction drawn per environment
(`oceanking.water`), the actor's observation extended by beta and the current in
the body's heading frame, and the foot-load rewards scaled to the net weight.
Fine-tuned from the official checkpoint (`oceanking/tools/expand_checkpoint.py`).

For evaluation, `--buoyancy`, `--current` and `--ground-friction` pin each
quantity for every environment instead of drawing it.
"""

from __future__ import annotations

from ...jumper.common.assets import JUMPER_ASSETS
from ...registry import register

register(
    id="ocean.walk",
    assets=JUMPER_ASSETS,
    description=(
        "jumper.posture adapted under water: beta, current and ground friction "
        "randomised per environment and observed (training task of docs/TRAINING.md)"
    ),
    tags=("locomotion", "jumper", "ocean", "posture"),
)


def cli_args(group):
    """Pins for evaluation (each falls back to an `OCEAN_*` variable); unset, the
    quantity is drawn from the training range."""
    import os

    def env(name):
        return os.environ.get(name, "").strip() or None

    group.add_argument("--buoyancy", dest="buoyancy_ratio", type=float,
                       default=env("OCEAN_BUOYANCY"), metavar="BETA",
                       help="pin beta for every environment (default: drawn 0.75-0.90)")
    current = env("OCEAN_CURRENT")
    group.add_argument("--current", dest="current", type=float, nargs=2,
                       default=[float(v) for v in current.split()] if current else None,
                       metavar=("VX", "VY"),
                       help="pin the current, world frame, m/s (default: drawn, 0-0.2 m/s)")
    group.add_argument("--ground-friction", dest="ground_friction", type=float,
                       default=env("OCEAN_GROUND_FRICTION"), metavar="MU",
                       help="pin the ground's friction (default: drawn 0.4-0.8)")
    group.add_argument("--nominal", dest="nominal", action="store_true",
                       default=(env("OCEAN_NOMINAL") or "") in ("1", "true", "on", "yes"),
                       help="replays: start at the origin facing +x, no startup "
                       "randomisation (see ocean.posture)")
    group.add_argument("--trajectory-out", dest="trajectory_out",
                       default=env("OCEAN_TRAJECTORY_OUT"), metavar="PATH",
                       help="replays: record root pose and command every control step")
    return ("buoyancy_ratio", "current", "ground_friction", "nominal", "trajectory_out")
