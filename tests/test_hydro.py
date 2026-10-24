"""Analytic checks of `oceanking.hydro` on both of Jumper's simulators.

Each test pins a failure that would otherwise be silent: buoyancy that is not
applied (or applied twice), drag computed against the absolute rather than the
relative velocity, a fluid model that quietly adds its own buoyancy, and water
lost when a scene is layered on top of a task.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from oceanking.hydro import HydroCfg, apply_to_spec

G = 9.81
BOX_HALF = (0.10, 0.06, 0.04)
BOX_MASS = 1.5


def _box_spec(*, floor: bool = False, z: float = 1.0) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if floor:
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
    body = spec.worldbody.add_body(name="box", pos=[0, 0, z])
    body.add_freejoint()
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=list(BOX_HALF), mass=BOX_MASS,
        contype=1 if floor else 0, conaffinity=1 if floor else 0,
    )
    return spec


def _model(cfg: HydroCfg | None, **kw) -> mujoco.MjModel:
    spec = _box_spec(**kw)
    if cfg is not None:
        apply_to_spec(spec, cfg)
    return spec.compile()


def _run_native(m: mujoco.MjModel, steps: int, qvel0=None) -> mujoco.MjData:
    d = mujoco.MjData(m)
    if qvel0 is not None:
        d.qvel[:] = qvel0
    for _ in range(steps):
        mujoco.mj_step(m, d)
    return d


def _run_warp(m: mujoco.MjModel, steps: int, qvel0=None) -> np.ndarray:
    """qvel after `steps` on MuJoCo Warp, one world."""
    wp = pytest.importorskip("warp")
    mjwarp = pytest.importorskip("mujoco_warp")
    d0 = mujoco.MjData(m)
    if qvel0 is not None:
        d0.qvel[:] = qvel0
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    with wp.ScopedDevice(device):
        wm = mjwarp.put_model(m)
        wd = mjwarp.put_data(m, d0)
        for _ in range(steps):
            mjwarp.step(wm, wd)
        return wd.qvel.numpy()[0]


BACKENDS = ["native", "warp"]


def _qvel(backend: str, m: mujoco.MjModel, steps: int, qvel0=None) -> np.ndarray:
    if backend == "native":
        return _run_native(m, steps, qvel0).qvel.copy()
    return _run_warp(m, steps, qvel0)


# ── buoyancy ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("beta", [0.0, 0.25, 0.5, 0.9])
def test_free_body_sinks_at_g_times_one_minus_beta(backend, beta):
    """An isolated body with no drag accelerates at -g (1 - beta)."""
    m = _model(HydroCfg(mode="buoyancy", buoyancy_ratio=beta))
    t = 0.2
    v = _qvel(backend, m, round(t / m.opt.timestep))
    np.testing.assert_allclose(v[2], -G * (1.0 - beta) * t, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_neutral_buoyancy_does_not_drift(backend):
    m = _model(HydroCfg(mode="buoyancy", buoyancy_ratio=1.0))
    v = _qvel(backend, m, 500)
    assert np.abs(v).max() < 1e-9


def test_mass_and_gravity_are_untouched():
    """Buoyancy must not be imitated by lighter bodies or weaker gravity."""
    land = _model(None)
    water = _model(HydroCfg(mode="full", buoyancy_ratio=0.5))
    np.testing.assert_array_equal(land.body_mass, water.body_mass)
    np.testing.assert_array_equal(land.body_inertia, water.body_inertia)
    np.testing.assert_array_equal(land.opt.gravity, water.opt.gravity)


def test_a_body_with_its_own_gravcomp_is_left_alone():
    """A prop that declares its buoyancy keeps it rather than being overwritten."""
    spec = _box_spec()
    spec.body("box").gravcomp = 0.8
    apply_to_spec(spec, HydroCfg(mode="buoyancy", buoyancy_ratio=0.3))
    m = spec.compile()
    assert m.body_gravcomp[m.body("box").id] == pytest.approx(0.8)


def test_standing_contact_load_is_weight_minus_buoyancy():
    """Resting on the floor, the support force is (1 - beta) m g. Native only:
    MuJoCo Warp's handling of gravcomp is pinned by the free-fall test above."""
    beta = 0.6
    m = _model(HydroCfg(mode="buoyancy", buoyancy_ratio=beta), floor=True, z=BOX_HALF[2])
    d = _run_native(m, 300)
    assert d.ncon > 0
    normal = 0.0
    for i in range(d.ncon):
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        normal += f[0]
    assert normal == pytest.approx((1.0 - beta) * BOX_MASS * G, rel=1e-3)
    # Control: on land the same box presses with its full weight.
    land_m = _model(None, floor=True, z=BOX_HALF[2])
    land_d = _run_native(land_m, 300)
    land_normal = 0.0
    for i in range(land_d.ncon):
        f = np.zeros(6)
        mujoco.mj_contactForce(land_m, land_d, i, f)
        land_normal += f[0]
    assert land_normal == pytest.approx(BOX_MASS * G, rel=1e-3)


# ── drag and current ────────────────────────────────────────────────────────


def test_the_fluid_model_adds_no_buoyancy():
    """Density alone (no gravcomp) leaves a body at rest falling at g, so drag and
    gravcomp never count the displaced volume twice."""
    m = _model(HydroCfg(mode="drag", water_density=1000.0))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert d.qacc[2] == pytest.approx(-G, rel=1e-9)
    assert np.abs(d.qfrc_passive).max() < 1e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_body_at_rest_in_a_current_is_pushed_downstream(backend):
    current = (0.3, -0.1, 0.0)
    m = _model(HydroCfg(mode="drag", current=current))
    m.opt.gravity[:] = 0.0
    v = _qvel(backend, m, 50)
    lin = v[:3]
    assert lin[0] > 0.0 and lin[1] < 0.0
    # The push is along the current (the box is axis-aligned, so drag is
    # diagonal in the world frame).
    assert abs(lin[2]) < 1e-9


@pytest.mark.parametrize("backend", BACKENDS)
def test_moving_with_the_current_feels_no_drag(backend):
    current = (0.3, 0.0, 0.0)
    m = _model(HydroCfg(mode="drag", current=current))
    m.opt.gravity[:] = 0.0
    qvel0 = np.zeros(6)
    qvel0[:3] = current
    v = _qvel(backend, m, 200, qvel0)
    np.testing.assert_allclose(v[:3], current, atol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_drag_only_removes_energy_relative_to_the_fluid(backend):
    """F_D . v_rel <= 0: in still water a moving body only slows down, and the
    control (no drag) keeps its speed, so the check can fail."""
    qvel0 = np.array([0.5, -0.2, 0.1, 0.0, 0.0, 0.0])
    still = _model(HydroCfg(mode="drag"))
    still.opt.gravity[:] = 0.0
    v = _qvel(backend, still, 200, qvel0)
    assert np.linalg.norm(v[:3]) < np.linalg.norm(qvel0[:3])
    assert np.all(np.sign(v[:3]) == np.sign(qvel0[:3]))

    control = _model(None)
    control.opt.gravity[:] = 0.0
    v = _qvel(backend, control, 200, qvel0)
    np.testing.assert_allclose(v[:3], qvel0[:3], atol=1e-9)


def test_drag_scale_changes_drag_and_not_buoyancy():
    a = _model(HydroCfg(mode="full", drag_scale=1.0))
    b = _model(HydroCfg(mode="full", drag_scale=3.0))
    assert b.opt.density == pytest.approx(3.0 * a.opt.density)
    np.testing.assert_array_equal(a.body_gravcomp, b.body_gravcomp)


@pytest.mark.parametrize("backend", BACKENDS)
def test_halving_the_timestep_agrees(backend):
    """Sink plus drag converges in the timestep: the finding does not depend on it."""
    out = []
    for dt in (0.002, 0.001):
        m = _model(HydroCfg(mode="full", buoyancy_ratio=0.5, current=(0.2, 0.0, 0.0)))
        m.opt.timestep = dt
        out.append(_qvel(backend, m, round(0.5 / dt))[:3])
    np.testing.assert_allclose(out[0], out[1], rtol=2e-3, atol=1e-4)


# ── modes ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mode,gravcomp,density",
    [("off", 0.0, 0.0), ("buoyancy", 0.5, 0.0), ("drag", 0.0, 1000.0), ("full", 0.5, 1000.0)],
)
def test_each_mode_sets_exactly_its_terms(mode, gravcomp, density):
    m = _model(HydroCfg(mode=mode, buoyancy_ratio=0.5, current=(0.1, 0.0, 0.0)))
    assert m.body_gravcomp[m.body("box").id] == gravcomp
    assert m.opt.density == density
    assert (m.opt.wind[0] == 0.1) == (density > 0)


def test_off_reproduces_land_exactly():
    land = _model(None)
    off = _model(HydroCfg(mode="off", buoyancy_ratio=0.7, current=(1.0, 0.0, 0.0)))
    a = _run_native(land, 300, np.array([0.1, 0, 0, 0, 0.2, 0]))
    b = _run_native(off, 300, np.array([0.1, 0, 0, 0, 0.2, 0]))
    np.testing.assert_array_equal(a.qpos, b.qpos)


def test_invalid_configuration_raises():
    with pytest.raises(ValueError):
        HydroCfg(mode="underwater")
    with pytest.raises(ValueError):
        HydroCfg(buoyancy_ratio=-0.1)
