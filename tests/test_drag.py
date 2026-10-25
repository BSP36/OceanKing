"""The drag calibration: plausible, current, and actually applied.

Silent failures pinned here: a calibration left over from another model or crown,
a calibration measured and then not passed to the simulator, and a calibrated
drag that has drifted out of the range a body of this kind can have.
"""

from __future__ import annotations

import json

import pytest

from oceanking import drag
from oceanking.hydro import HydroCfg


def test_the_calibration_is_for_this_model_and_crown():
    assert drag.calibration_factor() == pytest.approx(0.3226, abs=0.001)


def test_a_stale_calibration_is_refused(tmp_path, monkeypatch):
    table = json.loads(drag.DATA.read_text())
    table["fingerprint"]["crown_outer_volume_m3"] += 1e-4
    stale = tmp_path / "jumper_drag.json"
    stale.write_text(json.dumps(table))
    monkeypatch.setattr(drag, "DATA", stale)
    drag._table.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="another model or crown"):
            drag.calibration_factor()
    finally:
        drag._table.cache_clear()


@pytest.mark.parametrize("axis", ["head_on_x", "side_on_y", "heave_z"])
def test_calibrated_drag_coefficients_are_plausible(axis):
    """Rough bluff bodies at Re ~ 1e5: 0.8-1.3 on the silhouette (docs/PHYSICS.md).
    Control: the raw per-geom sum is far outside it."""
    entry = json.loads(drag.DATA.read_text())[axis]
    assert 0.8 <= entry["calibrated_Cd"] <= 1.3
    assert entry["raw_Cd"] > 2.5


def test_the_measurement_reproduces_the_stored_calibration():
    from oceanking.tools.link_volumes import XML

    fresh = drag.measure(XML)
    assert fresh["calibration_factor"] == pytest.approx(drag.calibration_factor(), rel=1e-6)


def test_the_calibration_reaches_the_model():
    import mujoco

    from oceanking.hydro import apply_to_spec
    from oceanking.tools.link_volumes import XML

    spec = mujoco.MjSpec.from_file(str(XML))
    apply_to_spec(spec, HydroCfg(mode="drag", drag_scale=2.0))
    m = spec.compile()
    assert m.opt.density == pytest.approx(1000.0 * 2.0 * drag.calibration_factor())
