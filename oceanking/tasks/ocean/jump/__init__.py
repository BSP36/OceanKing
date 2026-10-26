"""ocean.jump -- jumper.ref_free_jump, aimed, under water.

Training task of `docs/JUMP.md`: upstream's reference-free high jump with the
water of `oceanking.hydro`, beta (0.75-0.98) and ground friction drawn per
environment, and a target landing point per episode -- a strength (0 straight
up, 1 a full-reach jump) and a direction -- that the actor observes with beta.
Fine-tuned from the official checkpoint (`oceanking/tools/expand_checkpoint.py
--task jump`). For replays, `--buoyancy`, `--ground-friction` and
`--jump STRENGTH DEG` pin each instead of drawing it.
"""

from __future__ import annotations

from ...jumper.common.assets import JUMPER_ASSETS
from ...registry import register

register(
    id="ocean.jump",
    assets=JUMPER_ASSETS,
    description=(
        "jumper.ref_free_jump aimed at a landing point under water: strength and "
        "direction commanded, beta and ground friction randomised (docs/JUMP.md)"
    ),
    tags=("jumper", "jump", "ocean"),
)


def cli_args(group):
    """Pins for replays (each falls back to an `OCEAN_*` variable)."""
    import os

    def env(name):
        return os.environ.get(name, "").strip() or None

    group.add_argument("--buoyancy", dest="buoyancy_ratio", type=float,
                       default=env("OCEAN_BUOYANCY"), metavar="BETA",
                       help="pin beta for every environment (default: drawn 0.75-0.98)")
    group.add_argument("--ground-friction", dest="ground_friction", type=float,
                       default=env("OCEAN_GROUND_FRICTION"), metavar="MU",
                       help="pin the ground's friction (default: drawn 0.4-0.8)")
    jump = env("OCEAN_JUMP")
    group.add_argument("--jump", dest="jump", type=float, nargs=2,
                       default=[float(v) for v in jump.split()] if jump else None,
                       metavar=("STRENGTH", "DEG"),
                       help="pin the target: strength 0-1 (x reach 0.3 m) and direction, "
                       "degrees from forward, counter-clockwise (default: drawn)")
    group.add_argument("--nominal", dest="nominal", action="store_true",
                       default=(env("OCEAN_NOMINAL") or "") in ("1", "true", "on", "yes"),
                       help="replays: no startup randomisation (foot friction, COM, "
                       "encoder bias)")
    group.add_argument("--trajectory-out", dest="trajectory_out",
                       default=env("OCEAN_TRAJECTORY_OUT"), metavar="PATH",
                       help="replays: record the root pose every control step")
    return ("buoyancy_ratio", "ground_friction", "jump", "nominal", "trajectory_out")
