"""ocean.jump's aimed command, its observation columns and its rewards.

Silent failures pinned here: new columns that are not last (the expanded
checkpoint would read them in the wrong places), targets outside the reach or
an upright share that never happens, a jump declared over mid-flight (upstream's
`back_home` would end the episode in the air), and a landing reward or drift
anchor that reads the wrong point.
"""

from __future__ import annotations

import math

import pytest
import torch

tasks = pytest.importorskip("tasks")
pytest.importorskip("mjlab")

N = 64


@pytest.fixture(scope="module")
def env():
    from mjrl.backend import resolve
    from mjrl.backend.select import use_backend

    use_backend(resolve(backend="native", device="cpu", num_envs=N))
    from mjlab.envs import ManagerBasedRlEnv

    cfg = tasks.load_env_cfg("ocean.jump", play=False, task_args={})
    cfg.scene.num_envs = N
    e = ManagerBasedRlEnv(cfg, device="cpu")
    e.reset()
    yield e
    e.close()


def test_new_columns_are_last_and_sized_as_the_expanded_checkpoint(env):
    obs = env.observation_manager.compute()
    assert obs["actor"].shape[1] == 406 + 4
    assert obs["critic"].shape[1] == 435 + 8
    cmd = env.command_manager.get_term("jump")
    from tasks.ocean.jump import mdp

    tail = obs["actor"][:, -4:-1]
    assert torch.allclose(tail, mdp.jump_target(env), atol=1e-5)
    # Spawned facing +x: the heading-frame target is the world one over the reach.
    assert torch.allclose(tail[:, :2], cmd.target / cmd.cfg.reach, atol=1e-3)


def test_targets_lie_within_the_reach_with_an_upright_share(env):
    cmd = env.command_manager.get_term("jump")
    ids = torch.arange(N)
    upright, total = 0, 0
    for _ in range(20):
        cmd.can_jump[:] = True
        cmd._resample_command(ids)
        r = torch.linalg.norm(cmd.target, dim=1)
        assert (r <= cmd.cfg.reach + 1e-6).all()
        assert torch.allclose(r, cmd.strength * cmd.cfg.reach, atol=1e-6)
        upright += int((cmd.strength == 0).sum())
        total += N
    assert 0.1 < upright / total < 0.3


def test_a_jump_is_never_over_in_flight(env):
    cmd = env.command_manager.get_term("jump")
    env.episode_length_buf[:] = int(5.0 / env.step_dt)  # well past the recording's span
    cmd.go_step[:] = 0
    cmd.landed[:] = False
    cmd.in_flight[:] = True
    assert not cmd.jump_done.any()
    cmd.in_flight[:] = False
    assert cmd.jump_done.all()


def test_landing_reward_and_drift_anchor(env):
    from tasks.ocean.jump import mdp

    cmd = env.command_manager.get_term("jump")
    p = cmd.position()
    cmd.target[:] = p + torch.tensor([0.1, 0.0])
    cmd.second_jump[:] = False
    cmd.in_flight[:] = False
    # Not landed: nothing paid; drift measured from the take-off point (origin).
    cmd.landed[:] = False
    assert (mdp.landing_target(env, sigma=0.08) == 0).all()
    assert torch.allclose(mdp.anchored_drift(env, scale=0.1), p.square().sum(1) / 0.01)
    # Landed on the target: full landing reward; drift now from the landing point.
    cmd.landed[:] = True
    cmd.landing[:] = cmd.target
    assert torch.allclose(mdp.landing_target(env, sigma=0.08), torch.ones(N))
    assert torch.allclose(mdp.anchored_drift(env, scale=0.1), torch.full((N,), 1.0), atol=1e-4)
    # One sigma off: exp(-1). In flight: no drift.
    cmd.landing[:] = cmd.target + torch.tensor([0.0, 0.08])
    assert torch.allclose(mdp.landing_target(env, sigma=0.08), torch.full((N,), math.exp(-1)))
    cmd.in_flight[:] = True
    assert (mdp.anchored_drift(env, scale=0.1) == 0).all()
