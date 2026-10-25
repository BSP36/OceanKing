"""MuJoCo Warp's fluid torques agree with native MuJoCo once patched.

The silent failure: MuJoCo Warp 3.11.0 drops the moment of each geom's fluid
force about its body's centre of mass, so training (Warp) and replay (native)
ran different hydrodynamics with forces that matched and torques that did not.
Control: the unpatched launcher disagrees, so the comparison can fail.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

wp = pytest.importorskip("warp")
mjwarp = pytest.importorskip("mujoco_warp")

from oceanking import warp_patch  # noqa: E402
from oceanking.hydro import HydroCfg, apply_to_spec  # noqa: E402

XML = Path(__file__).resolve().parents[1] / "jumper" / "assets" / "jumper" / "jumper.xml"


def _offset_geom_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.option.density = 1000.0
    spec.option.gravity = [0, 0, 0]
    body = spec.worldbody.add_body(name="b")
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], mass=1.0,
                  contype=0, conaffinity=0)
    g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.02, 0.03],
                      pos=[0.2, 0.0, 0.0], mass=1e-6, contype=0, conaffinity=0)
    g.fluid_ellipsoid = 1.0
    return spec.compile()


def _fluid_both(m: mujoco.MjModel, d: mujoco.MjData, launcher) -> tuple[np.ndarray, np.ndarray]:
    mujoco.mj_forward(m, d)
    native = d.qfrc_fluid.copy()
    from mujoco_warp._src import passive

    saved = passive._fluid
    passive._fluid = launcher
    try:
        wp.init()
        with wp.ScopedDevice("cuda:0" if wp.is_cuda_available() else "cpu"):
            wm, wd = mjwarp.put_model(m), mjwarp.put_data(m, d)
            mjwarp.forward(wm, wd)
            warp = wd.qfrc_fluid.numpy()[0]
    finally:
        passive._fluid = saved
    return native, warp


def test_an_offset_geom_gets_its_moment():
    m = _offset_geom_model()
    d = mujoco.MjData(m)
    d.qvel[:] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    native, fixed = _fluid_both(m, d, warp_patch.fluid)
    np.testing.assert_allclose(fixed, native, rtol=1e-4, atol=1e-6)
    assert abs(native[5]) > 0.1  # the moment the bug drops
    _, broken = _fluid_both(m, d, warp_patch.original)
    assert abs(broken[5] - native[5]) > 0.1


def test_the_kitted_robot_agrees_across_simulators():
    spec = mujoco.MjSpec.from_file(str(XML))
    apply_to_spec(spec, HydroCfg(mode="full", buoyancy_ratio=0.8, current=(-0.3, 0.2, 0.0)))
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[2] = 0.12
    rng = np.random.default_rng(0)
    d.qvel[:] = rng.normal(scale=1.0, size=m.nv)
    native, fixed = _fluid_both(m, d, warp_patch.fluid)
    np.testing.assert_allclose(fixed, native, rtol=2e-3, atol=2e-5)
    _, broken = _fluid_both(m, d, warp_patch.original)
    assert np.abs(broken[3:6] - native[3:6]).max() > 1e-3


def test_the_patch_is_installed_by_hydro():
    from mujoco_warp._src import passive

    assert passive._fluid is warp_patch.fluid
