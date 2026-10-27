"""The crown tank changes what it must, while the simulation runs.

Silent failures pinned here: a tank whose water never reaches the crown body's
mass (the observation would claim one beta and the physics run another), a beta
in the observation computed against the wrong keel, and a pump that moves the
water instantly.
"""

from __future__ import annotations

import pytest
import torch

tasks = pytest.importorskip("tasks")
pytest.importorskip("mjlab")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from mjrl.backend import resolve
    from mjrl.backend.select import use_backend

    script = tmp_path_factory.mktemp("tank") / "blow.py"
    script.write_text("def air(t):\n    return 1.0\n")
    use_backend(resolve(backend="native", device="cpu", num_envs=2))
    from mjlab.envs import ManagerBasedRlEnv

    cfg = tasks.load_env_cfg("ocean.jump", play=True,
                             task_args={"tank_script": str(script), "jump": [0.0, 0.0]})
    cfg.scene.num_envs = 2
    e = ManagerBasedRlEnv(cfg, device="cpu")
    e.reset()
    yield e
    e.close()


def test_the_pump_moves_water_at_its_rate_and_beta_follows(env):
    from oceanking.tank import CROWN
    from tasks.ocean.jump.env_cfg import SUBMARINE, TANK_RATE
    from oceanking.water import ocean_state

    tank = env.crown_tank
    crown = env.sim.mj_model.body(CROWN).id
    zero = torch.zeros(2, env.action_manager.total_action_dim)
    env.step(zero)
    assert tank.air[0].item() == pytest.approx(TANK_RATE * env.step_dt, abs=1e-6)
    for _ in range(int(1.0 / env.step_dt)):
        env.step(zero)
    a = tank.air[0].item()
    assert 0.4 < a < 0.6  # half a tank after ~1 s at 0.5 per second
    water = 1000.0 * 0.803e-3 * (1 - a)
    shell = float(env.sim.model.body_mass[0, crown]) - water
    assert shell == pytest.approx(tank._shell, abs=2e-3)
    # The observed beta is the kit's at this air fraction (Ballast.beta_at, an
    # independent formula), between the flooded and the blown figures.
    from oceanking.hydro import HydroCfg

    kit = HydroCfg(**SUBMARINE).ballast()
    assert ocean_state(env).beta[0].item() == pytest.approx(kit.beta_at(a), rel=1e-4)
    assert kit.beta_at(0.0) < ocean_state(env).beta[0].item() < kit.beta_at(1.0)
