"""Measure the drag calibration and write `oceanking/data/jumper_drag.json`.

See `oceanking/drag.py`. Re-run whenever the robot model or the crown changes;
`oceanking.drag` refuses a stale file.

    .venv/bin/python -m oceanking.tools.drag_calibration
"""

from __future__ import annotations

import json

from oceanking import drag
from oceanking.tools.link_volumes import XML


def main() -> None:
    table = {"fingerprint": drag._fingerprint(), "cd_target_head_on": drag.CD_TARGET,
             "speed_m_s": drag.SPEED, **drag.measure(XML)}
    drag.DATA.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    for k in ("head_on_x", "side_on_y", "heave_z"):
        r = table[k]
        print(f"{k:10s} silhouette {r['silhouette_m2'] * 1e4:6.0f} cm2  raw Cd {r['raw_Cd']:.2f}"
              f"  calibrated Cd {r['calibrated_Cd']:.2f}")
    print(f"calibration factor {table['calibration_factor']:.4f} -> {drag.DATA}")


if __name__ == "__main__":
    main()
