"""Archimedes buoyancy and the ballast kit on the real Jumper model.

Silent failures pinned here: a keel solved against the wrong totals (the robot
then sits at another beta than the one reported), a volume table applied to a
model it was not measured on, buoyancy that ignores where volume and mass are
(no righting moment), and a kit that changes the land robot.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from oceanking import buoyancy as B
from oceanking.hydro import HydroCfg, apply_to_spec

REPO = Path(__file__).resolve().parents[1]
XML = REPO / "jumper" / "assets" / "jumper" / "jumper.xml"
G = 9.81
KP, KD, TAU = 10.0, 0.5, 1.7464  # jumper.posture's PD gains and servo plateau


def _robot(cfg: HydroCfg | None, *, floor: bool = True) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(XML))
    if floor:
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
    if cfg is not None:
        apply_to_spec(spec, cfg)
    return spec.compile()


def _home(m: mujoco.MjModel, z: float = 0.12) -> mujoco.MjData:
    from tasks.jumper.common.constants import HOME

    d = mujoco.MjData(m)
    d.qpos[2] = z
    for name, q in HOME.items():
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = q
    return d


def test_the_table_is_for_this_model():
    B.check_model(XML)
    table = B.link_table()
    m = _robot(None, floor=False)
    assert table["total_mass_kg"] == pytest.approx(m.body_mass.sum(), rel=1e-9)


def test_a_different_model_is_refused(tmp_path):
    other = tmp_path / "jumper.xml"
    other.write_bytes(XML.read_bytes() + b"\n<!-- edited -->\n")
    with pytest.raises(RuntimeError, match="not the model"):
        B.check_model(other)


def test_sealed_jumper_floats():
    """No kit: 3.42 L against 2.543 kg. The number the whole kit exists for."""
    table = B.link_table()
    assert table["sealed_buoyancy_ratio_fresh_water"] == pytest.approx(1.346, abs=0.005)


@pytest.mark.parametrize("beta,air", [(0.8, 0.5), (0.9, 0.0), (0.95, 1.0)])
def test_the_compiled_robot_has_the_requested_beta(beta, air):
    cfg = HydroCfg(mode="buoyancy", buoyancy_ratio=beta, crown_air=air)
    m = _robot(cfg, floor=False)
    assert B.whole_robot_beta(m, "base_link") == pytest.approx(beta, rel=1e-9)
    ballast = cfg.ballast()
    assert m.body_mass.sum() == pytest.approx(ballast.total_mass, rel=1e-9)
    assert ballast.beta_at(air) == pytest.approx(beta, rel=1e-12)


def test_each_link_carries_its_own_volume():
    cfg = HydroCfg(mode="buoyancy", buoyancy_ratio=0.8)
    m = _robot(cfg, floor=False)
    for name, entry in B.link_table()["links"].items():
        i = m.body(name).id
        assert m.body_mass[i] * m.body_gravcomp[i] == pytest.approx(
            1000.0 * entry["volume_m3"], rel=1e-9, abs=1e-12
        ), name


def test_the_keel_adds_mass_and_no_buoyancy_and_the_crown_floats():
    m = _robot(HydroCfg(mode="buoyancy", buoyancy_ratio=0.8), floor=False)
    keel, crown = m.body("oceanking_keel").id, m.body("oceanking_crown").id
    assert m.body_gravcomp[keel] == 0.0
    assert m.body_gravcomp[crown] > 1.0
    assert m.body_mass[keel] == pytest.approx(HydroCfg(buoyancy_ratio=0.8).ballast().keel_mass)


@pytest.mark.parametrize("beta", [0.8, 0.95])
def test_standing_load_is_weight_minus_buoyancy(beta):
    """Held at HOME by the policy's PD gains, the ground carries (1 - beta) M g."""
    m = _robot(HydroCfg(mode="buoyancy", buoyancy_ratio=beta))
    d = _home(m)
    qadr = [m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)]
    vadr = [m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)]
    target = d.qpos[qadr].copy()
    loads = []
    for k in range(3000):
        d.ctrl[:] = np.clip(KP * (target - d.qpos[qadr]) - KD * d.qvel[vadr], -TAU, TAU)
        mujoco.mj_step(m, d)
        if k >= 2000:
            total = 0.0
            for i in range(d.ncon):
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f)
                total += f[0]
            loads.append(total)
    weight = m.body_mass.sum() * G
    # Tolerance against the full weight: near neutral the net load is small, and
    # the residual sway of a PD-held stance (a few mN) is not a buoyancy error.
    assert np.mean(loads) == pytest.approx((1.0 - beta) * weight, abs=2e-3 * weight)


def _root_torque(m: mujoco.MjModel, roll: float) -> np.ndarray:
    """Net gravity + buoyancy torque on the free root, robot at rest and rolled."""
    d = _home(m, z=1.0)
    d.qpos[3:7] = [math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]
    mujoco.mj_forward(m, d)
    return (d.qfrc_passive - d.qfrc_bias)[3:6]


def test_the_kit_rights_a_rolled_robot():
    """Keel low and crown high put the centre of buoyancy above the centre of
    mass: rolled left, the robot is turned back right. Control: the uniform model
    puts every buoyant force at a centre of mass, so it has no righting moment."""
    kit = _robot(HydroCfg(mode="buoyancy", buoyancy_ratio=0.95), floor=False)
    tau = _root_torque(kit, math.radians(30))
    assert tau[0] < -0.05  # N*m, restoring about x

    uniform = _robot(HydroCfg(mode="buoyancy", buoyancy_model="uniform",
                              buoyancy_ratio=0.95), floor=False)
    tau_u = _root_torque(uniform, math.radians(30))
    land = _root_torque(_robot(None, floor=False), math.radians(30))
    np.testing.assert_allclose(tau_u, 0.05 * land, atol=1e-9)  # (1 - beta) of the land torque


def test_a_keel_that_does_not_fit_is_refused():
    with pytest.raises(ValueError, match="torso"):
        HydroCfg(buoyancy_ratio=0.5).ballast()


def test_negative_ballast_is_refused():
    with pytest.raises(ValueError, match="negative ballast"):
        HydroCfg(buoyancy_ratio=2.0, crown_air=1.0).ballast()


def test_off_and_drag_only_leave_the_robot_as_upstream_built_it():
    land = _robot(None)
    for mode in ("off", "drag"):
        m = _robot(HydroCfg(mode=mode, buoyancy_ratio=0.8))
        assert m.nbody == land.nbody
        np.testing.assert_array_equal(m.body_mass, land.body_mass)
        np.testing.assert_array_equal(m.body_gravcomp, land.body_gravcomp)


@pytest.mark.parametrize("backend", ["native", "warp"])
def test_whole_robot_sinks_at_g_times_one_minus_beta(backend):
    """Gravity and buoyancy are the only external forces in free water without
    drag, so the centre of mass of the whole robot, kit included, accelerates at
    -(1 - beta) g however the limbs move. Checked on both simulators."""
    beta, t = 0.85, 0.1
    m = _robot(HydroCfg(mode="buoyancy", buoyancy_ratio=beta), floor=False)
    d = _home(m, z=1.0)
    steps = round(t / m.opt.timestep)
    if backend == "native":
        for _ in range(steps):
            mujoco.mj_step(m, d)
    else:
        wp = pytest.importorskip("warp")
        mjwarp = pytest.importorskip("mujoco_warp")
        wp.init()
        with wp.ScopedDevice("cuda:0" if wp.is_cuda_available() else "cpu"):
            wm, wd = mjwarp.put_model(m), mjwarp.put_data(m, d)
            for _ in range(steps):
                mjwarp.step(wm, wd)
            d.qpos[:], d.qvel[:] = wd.qpos.numpy()[0], wd.qvel.numpy()[0]
    mujoco.mj_forward(m, d)
    mujoco.mj_subtreeVel(m, d)
    com_vz = d.subtree_linvel[m.body("base_link").id][2]
    assert com_vz == pytest.approx(-(1.0 - beta) * G * t, rel=1e-3)
