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


N_LINKS = len(B.link_table()["links"])


@pytest.mark.parametrize("beta,air", [(0.8, 0.5), (0.9, 0.0), (0.95, 1.0)])
def test_the_compiled_robot_has_the_requested_beta(beta, air):
    """Weight minus buoyancy of the compiled model is (1 - beta) M g, M the
    physical mass (robot, keel, crown with its water)."""
    cfg = HydroCfg(mode="buoyancy", buoyancy_ratio=beta, crown_air=air)
    m = _robot(cfg, floor=False)
    ballast = cfg.ballast()
    assert B.net_weight(m, "base_link") == pytest.approx(
        (1.0 - beta) * ballast.total_mass * G, rel=1e-6)
    assert m.body_mass.sum() == pytest.approx(
        ballast.total_mass + N_LINKS * B.COB_MASS, rel=1e-9)
    assert ballast.beta_at(air) == pytest.approx(beta, rel=1e-12)


def test_each_link_is_lifted_at_its_own_centre_of_buoyancy():
    """Each link's buoyancy acts at the centroid of its measured volume, not at
    its centre of mass, and lifts exactly rho V; the link itself carries none."""
    m = _robot(HydroCfg(mode="buoyancy", buoyancy_ratio=0.8), floor=False)
    for name, entry in B.link_table()["links"].items():
        link, point = m.body(name).id, m.body(f"{name}_cob").id
        assert m.body_parentid[point] == link
        assert m.body_gravcomp[link] == 0.0
        lift = (m.body_gravcomp[point] - 1.0) * m.body_mass[point]
        assert lift == pytest.approx(1000.0 * entry["volume_m3"], rel=1e-9, abs=1e-12), name
        np.testing.assert_allclose(m.body_pos[point], entry["centre_of_buoyancy_body_m"],
                                   atol=1e-12)


def test_buoyancy_points_feel_negligible_drag_and_the_keel_none():
    """A massive body with no flagged geom falls back to MuJoCo's inertia-box drag
    model. The keel is flagged with zero coefficients; the buoyancy points are
    left on the inertia-box model, sized so that all of them together feel under
    0.1% of the robot's drag."""
    m = _robot(HydroCfg(mode="full", buoyancy_ratio=0.8), floor=False)
    fluid = m.geom_fluid.reshape(m.ngeom, -1)
    keel = m.body("oceanking_keel").id
    flagged = [g for g in range(m.ngeom) if m.geom_bodyid[g] == keel and fluid[g, 0] > 0]
    assert flagged and not any(fluid[g, 1:6].any() for g in flagged)

    from oceanking import drag

    rho, v = 1000.0, 1.0
    points = 0.0
    for b in range(m.nbody):
        if m.body(b).name.endswith("_cob"):
            i, mass = m.body_inertia[b], m.body_mass[b]
            box = [math.sqrt(max(i[(k + 1) % 3] + i[(k + 2) % 3] - i[k], 0) * 6 / mass)
                   for k in range(3)]
            points += 0.5 * rho * max(box[0] * box[1], box[1] * box[2], box[0] * box[2]) * v**2
    head = drag._table()["head_on_x"]
    robot = 0.5 * rho * head["calibrated_Cd"] * head["silhouette_m2"] * v**2
    assert points < 1e-3 * robot


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
    weight = HydroCfg(buoyancy_ratio=beta).ballast().total_mass * G
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


def test_off_leaves_the_robot_as_upstream_built_it():
    land = _robot(None)
    m = _robot(HydroCfg(mode="off", buoyancy_ratio=0.8))
    assert m.nbody == land.nbody
    np.testing.assert_array_equal(m.body_mass, land.body_mass)
    np.testing.assert_array_equal(m.body_gravcomp, land.body_gravcomp)


def test_added_mass_grows_inertia_and_not_weight():
    """Drag-only water: no kit, no buoyancy; each link gains C_a rho V of inertia
    whose weight is compensated, so the net weight stays the land weight."""
    land = _robot(None, floor=False)
    m = _robot(HydroCfg(mode="drag", added_mass_coef=0.5), floor=False)
    assert m.nbody == land.nbody
    for name, entry in B.link_table()["links"].items():
        i = m.body(name).id
        dm = 0.5 * 1000.0 * entry["volume_m3"]
        assert m.body_mass[i] == pytest.approx(land.body_mass[i] + dm, rel=1e-9)
        scale = m.body_mass[i] / land.body_mass[i]
        np.testing.assert_allclose(m.body_inertia[i], land.body_inertia[i] * scale, rtol=1e-9)
    assert B.net_weight(m, "base_link") == pytest.approx(land.body_mass.sum() * G, rel=1e-9)


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


def _upside_down_crown_bottom(collide: bool) -> tuple[float, int]:
    """Set the kitted robot upside down with its crown 2 mm above the floor and
    let it settle for 0.3 s; return the crown's lowest point and how many
    contacts the crown has.

    Short on purpose: balanced on the dome the robot topples within a second, and
    in water the kit turns it over (both measured 2026-10-24). Buoyancy is zeroed:
    this is a test of the crown's collision, not of its buoyancy."""
    spec = mujoco.MjSpec.from_file(str(XML))
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
    apply_to_spec(spec, HydroCfg(mode="buoyancy", buoyancy_ratio=0.8))
    for body in spec.bodies:
        body.gravcomp = 0.0
    if not collide:
        for g in spec.geoms:
            if g.name.startswith("oceanking_crown"):
                g.contype = g.conaffinity = 0
    m = spec.compile()
    orb = m.geom("oceanking_crown_orb").id  # the crown's highest part: lowest, inverted
    d = _home(m, z=1.0)
    d.qpos[3:7] = [0.0, 1.0, 0.0, 0.0]  # rolled 180 degrees: crown down
    mujoco.mj_forward(m, d)
    d.qpos[2] -= d.geom_xpos[orb][2] - m.geom_size[orb][0] - 0.002
    for _ in range(300):
        mujoco.mj_step(m, d)
    crown = {g for g in range(m.ngeom) if m.geom(g).name.startswith("oceanking_crown")}
    contacts = sum(bool({c.geom1, c.geom2} & crown) for c in d.contact[: d.ncon])
    bottom = min(d.geom_xpos[g][2] for g in crown) - m.geom_size[orb][0]
    return bottom, contacts


def test_an_upside_down_robot_rests_on_its_crown():
    """The crown collides with the ground like the torso. Control: without its
    collision it sinks into the floor, so the check can fail."""
    bottom, contacts = _upside_down_crown_bottom(collide=True)
    assert contacts > 0 and bottom > -0.005
    bottom, contacts = _upside_down_crown_bottom(collide=False)
    assert contacts == 0 and bottom < -0.02


@pytest.mark.parametrize("backend", ["native", "warp"])
def test_added_mass_slows_the_sink(backend):
    """With added mass the same net weight accelerates more mass: the centre of
    mass sinks at (1 - beta) M g / (M + C_a rho V_total). Drag is zeroed so only
    added mass differs from the free-fall test above."""
    beta, ca, t = 0.85, 0.5, 0.1
    cfg = HydroCfg(mode="full", buoyancy_ratio=beta, drag_scale=0.0, added_mass_coef=ca)
    m = _robot(cfg, floor=False)
    ballast = cfg.ballast()
    added = ca * 1000.0 * ballast.total_volume
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
    expected = -(1.0 - beta) * ballast.total_mass * G / (
        ballast.total_mass + added + N_LINKS * B.COB_MASS) * t
    assert com_vz == pytest.approx(expected, rel=1e-3)
    # Control: it is slower than without added mass.
    assert abs(com_vz) < (1.0 - beta) * G * t * 0.8


def test_kit_and_buoyancy_points_have_their_intended_inertial_frames():
    """MjSpec leaves an explicit inertial's ipos unset (NaN) by default, and the
    compiler then puts the centre of mass at the body's position *again*: a point
    meant 49 mm out from its link landed 98 mm out (found 2026-10-25). Pin every
    body this package adds."""
    from oceanking import crown

    m = _robot(HydroCfg(mode="buoyancy", buoyancy_ratio=0.8), floor=False)
    for b in range(m.nbody):
        name = m.body(b).name
        if name.endswith("_cob") or name == "oceanking_keel":
            np.testing.assert_array_equal(m.body_ipos[b], 0.0, err_msg=name)
    np.testing.assert_allclose(m.body_ipos[m.body("oceanking_crown").id],
                               [0.0, 0.0, crown.centre_height()], atol=1e-12)


def test_the_keel_trims_the_robot_level():
    """At HOME the whole robot's centre of mass lies under its centre of
    buoyancy: no static pitch or roll moment. Control: the keel at the torso's
    centre leaves an offset of several millimetres."""
    from oceanking.tools.physics_report import _model

    def offset(cfg):
        m, d = _model(cfg)
        lift = np.zeros(m.nbody)
        for b in range(m.nbody):
            if m.body(b).name.endswith("_cob"):
                lift[b] = (m.body_gravcomp[b] - 1.0) * m.body_mass[b]
            elif m.body(b).name == "oceanking_crown":
                lift[b] = m.body_gravcomp[b] * m.body_mass[b]
        phys = np.array([not m.body(b).name.endswith("_cob") for b in range(m.nbody)])
        com = (m.body_mass[phys, None] * d.xipos[phys]).sum(0) / m.body_mass[phys].sum()
        cob = (lift[:, None] * d.xipos).sum(0) / lift.sum()
        return (cob - com)[:2]

    for beta in (0.7, 0.8, 0.95):
        np.testing.assert_allclose(offset(HydroCfg(mode="buoyancy", buoyancy_ratio=beta)),
                                   0.0, atol=1e-6)
