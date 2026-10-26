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
    assert torch.allclose(mdp.anchored_drift(env, scale=0.1, deadband=0.05),
                          torch.full((N,), 0.25), atol=1e-4)
    # One sigma off: exp(-1). In flight: no drift.
    cmd.landing[:] = cmd.target + torch.tensor([0.0, 0.08])
    assert torch.allclose(mdp.landing_target(env, sigma=0.08), torch.full((N,), math.exp(-1)))
    cmd.in_flight[:] = True
    assert (mdp.anchored_drift(env, scale=0.1) == 0).all()


def test_drift_before_take_off_has_a_deadband(env):
    from tasks.ocean.jump import mdp

    cmd = env.command_manager.get_term("jump")
    cmd.landed[:] = False
    cmd.in_flight[:] = False
    d = cmd.position().norm(dim=1)
    expect = (d - 0.05).clamp(min=0.0).square() / 0.01
    assert torch.allclose(mdp.anchored_drift(env, scale=0.1, deadband=0.05), expect)


def test_flight_progress_pays_speed_towards_the_target_in_flight_only(env):
    from tasks.ocean.jump import mdp

    cmd = env.command_manager.get_term("jump")
    p = cmd.position()
    cmd.target[:] = p + torch.tensor([0.2, 0.0])
    cmd.strength[:] = 0.7
    robot = env.scene["robot"]
    v = torch.zeros(N, 6)
    v[:, 0] = 0.4
    robot.write_root_link_velocity_to_sim(v)
    env.sim.forward()
    env.scene.update(0.0)
    cmd.in_flight[:] = False
    assert (mdp.flight_progress(env) == 0).all()
    cmd.in_flight[:] = True
    got = mdp.flight_progress(env)
    assert torch.allclose(got, torch.full((N,), 0.4), atol=0.02)
    cmd.strength[:] = 0.0
    assert (mdp.flight_progress(env) == 0).all()


def test_contact_threshold_follows_the_net_weight_and_ignores_the_rising_push(env):
    """Runs 1-3: with upstream's 1 N a robot at beta 0.9+ read as airborne while
    standing, and a touch during the push latched as the landing."""
    from tasks.ocean.jump import mdp
    from oceanking import water

    cmd = env.command_manager.get_term("jump")
    cmd.in_flight[:] = False
    cmd._update_command()
    thr = cmd.cfg.force_threshold.flatten()
    scale = water.ocean_state(env).load_scale
    assert torch.allclose(thr, (1.0 * scale).clamp_min(mdp.MIN_CONTACT_N))
    assert (thr < 1.0).all()
    assert cmd.cfg.min_rise == pytest.approx(0.04)
