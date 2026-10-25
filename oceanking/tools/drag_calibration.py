"""Measure the per-group drag calibration and write `oceanking/data/jumper_drag.json`.

See `oceanking/drag.py`. Re-run whenever the robot model, the crown or a target
changes; `oceanking.drag` refuses a stale file.

    .venv/bin/python -m oceanking.tools.drag_calibration
"""

from __future__ import annotations

import json

from oceanking import drag
from oceanking.tools.link_volumes import XML


def main() -> None:
    table = {"fingerprint": drag._fingerprint(), "speed_m_s": drag.SPEED, **drag.measure(XML)}
    drag.DATA.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    for name, e in table["groups"].items():
        print(f"group {name:6s} {e['geoms']:3d} geoms {len(e['units'])} units, reference area "
              f"{e['reference_area_m2'] * 1e4:5.0f} cm2,"
              f" raw Cd {e['raw_Cd']:.2f} -> target {e['target_Cd']:.2f}, factor {e['factor']:.3f}")
    for k in ("head_on_x", "side_on_y", "heave_z"):
        r = table[k]
        print(f"whole {k:10s} silhouette {r['silhouette_m2'] * 1e4:5.0f} cm2  raw Cd {r['raw_Cd']:.2f}"
              f"  calibrated Cd {r['calibrated_Cd']:.2f}")


if __name__ == "__main__":
    main()
