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


ARGS = dict(
    hydro_mode="full", buoyancy_ratio=0.85, buoyancy_model="archimedes", crown_air=0.3,
    current=(0.1, -0.05), drag_scale=2.0,
)


def _beta(m, args) -> float:
    """1 - net weight / physical weight, the physical mass from the solved kit."""
    from oceanking.buoyancy import net_weight
    from oceanking.hydro import HydroCfg

    mass = HydroCfg(buoyancy_ratio=args["buoyancy_ratio"],
                    crown_air=args["crown_air"]).ballast().total_mass
    return 1.0 - net_weight(m, "robot/base_link") / (mass * 9.81)


@pytest.mark.parametrize("scene", [None, "seabed", "studio"])
def test_task_arguments_reach_the_model_under_any_scene(scene):
    m = _compile(ARGS, scene)
    assert _beta(m, ARGS) == pytest.approx(0.85, rel=1e-6)
    assert m.body("robot/oceanking_crown").id > 0
    # Drag is on the robot's colliding geoms after mjlab set their contype.
    flagged = m.geom_fluid.reshape(m.ngeom, -1)[:, 0] > 0
    robot = np.array([m.geom(g).name.startswith("robot/") for g in range(m.ngeom)])
    assert flagged[robot].sum() >= 41
    from oceanking.drag import calibration_factor

    assert m.opt.density == pytest.approx(2000.0 * calibration_factor())
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
    water = _compile(ARGS)
    assert water.nbody != land.nbody and water.opt.density > 0.0
