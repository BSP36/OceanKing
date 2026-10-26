"""ocean.jump -- `jumper.ref_free_jump` on the seabed.

Upstream's reference-free high jump, unchanged in observation, action, reward and
episode, with `oceanking.hydro` installed on top, so the official checkpoint runs
here as it is: the baseline for an underwater jump.
"""

from __future__ import annotations

from ...jumper.common.assets import JUMPER_ASSETS
from ...registry import register
from ..posture import cli_args  # noqa: F401  (the same water arguments)

register(
    id="ocean.jump",
    assets=JUMPER_ASSETS,
    description=(
        "jumper.ref_free_jump underwater with the OceanKing ballast kit: buoyancy, drag "
        "and a current set by --hydro, --buoyancy, --crown-air and --current"
    ),
    tags=("jumper", "jump", "ocean"),
)
