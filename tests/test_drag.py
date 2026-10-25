"""The per-group drag calibration: current, reproducible, applied, and meaningful.

Silent failures pinned here: a calibration left over from another model, crown
or target; a calibration measured and not passed to the simulator; and one
factor for the whole robot, which cut the legs' drag to half a cylinder's
(review of 2026-10-25).
"""

from __future__ import annotations

import json

import pytest

from oceanking import drag
from oceanking.hydro import HydroCfg


def _stored() -> dict:
    return json.loads(drag.DATA.read_text())


def test_a_stale_calibration_is_refused(tmp_path, monkeypatch):
    table = _stored()
    table["fingerprint"]["crown_outer_volume_m3"] += 1e-4
    stale = tmp_path / "jumper_drag.json"
    stale.write_text(json.dumps(table))
    monkeypatch.setattr(drag, "DATA", stale)
    drag._table.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="another model"):
            drag.group_factors()
    finally:
        drag._table.cache_clear()


def test_the_measurement_reproduces_the_stored_calibration():
    from oceanking.tools.link_volumes import XML

    fresh = drag.measure(XML)
    for group, factor in drag.group_factors().items():
        assert fresh["groups"][group]["factor"] == pytest.approx(factor, rel=1e-6), group


def test_groups_are_calibrated_separately():
    """The point of groups: the torso's overlapping hulls and the crown's small
    parts are cut much harder than the legs, which raw are close to a cylinder.
    Control: a single factor would make these equal."""
    f = drag.group_factors()
    assert f["legs"] > 2.0 * f["torso"]
    assert set(f) == set(drag.CD_TARGET)
    groups = _stored()["groups"]
    assert 1.0 < groups["legs"]["raw_Cd"] < 2.0  # a near-cylinder before calibration


def test_whole_robot_drag_is_the_sum_of_its_calibrated_parts():
    """Head-on, the calibrated robot's drag area is the groups' target areas summed
    (no shielding between parts), to within MuJoCo's few-percent non-additivity."""
    t = _stored()
    expected = sum(g["target_Cd"] * g["reference_area_m2"] for g in t["groups"].values())
    whole = t["head_on_x"]["calibrated_Cd"] * t["head_on_x"]["silhouette_m2"]
    assert whole == pytest.approx(expected, rel=0.05)


def test_the_calibration_reaches_the_model():
    import mujoco

    from oceanking.hydro import apply_to_spec
    from oceanking.tools.link_volumes import XML

    spec = mujoco.MjSpec.from_file(str(XML))
    apply_to_spec(spec, HydroCfg(mode="full", buoyancy_ratio=0.8, drag_scale=2.0))
    m = spec.compile()
    assert m.opt.density == pytest.approx(2000.0)
    fluid = m.geom_fluid.reshape(m.ngeom, -1)
    factors = drag.group_factors()
    for g in range(m.ngeom):
        group = drag.group_of(m.body(m.geom_bodyid[g]).name)
        if group is not None and fluid[g, 0] > 0:
            assert fluid[g, 0] == pytest.approx(factors[group]), m.geom(g).name
    assert fluid[m.geom("oceanking_keel_plate").id, 0] < 1e-6
