"""The water reaches the compiled model through the real task and scene path.

The silent failures pinned here: a task argument parsed and never passed on, and
a scene layered on with `--scene` replacing the task's `spec_fn` so the robot is
back on land with nothing reporting it.
"""

from __future__ import annotations

import numpy as np
import pytest

tasks = pytest.importorskip("tasks")
scenes = pytest.importorskip("scenes")
pytest.importorskip("mjlab")


def _compile(task_args: dict, scene: str | None = None, **scene_args):
    from mjlab.scene import Scene

    cfg = tasks.load_env_cfg("ocean.posture", play=True, task_args=task_args)
    if scene is not None:
        scenes.apply(cfg, scene, **scene_args)
    return Scene(cfg.scene, device="cpu").compile()


ARGS = dict(hydro_mode="full", buoyancy_ratio=0.5, current=(0.1, -0.05), drag_scale=2.0)


def _robot_mass_weighted_beta(m) -> float:
    moving = m.body_dofnum > 0
    tree = np.isin(m.body_rootid, m.body_rootid[moving])
    tree[0] = False
    return float((m.body_mass * m.body_gravcomp)[tree].sum() / m.body_mass[tree].sum())


@pytest.mark.parametrize("scene", [None, "seabed", "studio"])
def test_task_arguments_reach_the_model_under_any_scene(scene):
    m = _compile(ARGS, scene)
    assert _robot_mass_weighted_beta(m) == pytest.approx(0.5)
    assert m.opt.density == pytest.approx(2000.0)
    np.testing.assert_allclose(m.opt.wind, (0.1, -0.05, 0.0))


def test_off_is_upstream_posture():
    """`ocean.posture --hydro off` compiles to the same physics as `jumper.posture`."""
    from mjlab.scene import Scene

    off = _compile(dict(ARGS, hydro_mode="off"))
    land = Scene(tasks.load_env_cfg("jumper.posture", play=True).scene, device="cpu").compile()
    np.testing.assert_array_equal(off.body_gravcomp, land.body_gravcomp)
    np.testing.assert_array_equal(off.body_mass, land.body_mass)
    assert off.opt.density == land.opt.density == 0.0
    # Control: the same comparison does see the water when it is on.
    assert _robot_mass_weighted_beta(_compile(ARGS)) != _robot_mass_weighted_beta(land)
