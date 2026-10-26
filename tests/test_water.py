"""ocean.walk's per-environment water reaches the simulator, on both backends.

Silent failures pinned here: a drawn beta that never reaches the keel's mass (the
observation would say one thing and the physics do another), a current written to
one world's array while every world reads it, a ground friction that the feet
outrank, an observation whose new columns are not where the expanded checkpoint
put them, a mirror that does not negate the current's left component, and a
current curriculum opened at once by mjlab's restored step counter.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

tasks = pytest.importorskip("tasks")
scenes = pytest.importorskip("scenes")
pytest.importorskip("mjlab")

N = 6
G = 9.81


def _build(backend: str):
    from mjrl.backend import resolve
    from mjrl.backend.select import use_backend

    if backend == "warp":
        wp = pytest.importorskip("warp")
        wp.init()
        if not wp.is_cuda_available():
            pytest.skip("MuJoCo Warp needs CUDA here")
    res = resolve(backend=backend, device="cuda:0" if backend == "warp" else "cpu", num_envs=N)
    use_backend(res)
    from mjlab.envs import ManagerBasedRlEnv

    cfg = tasks.load_env_cfg("ocean.walk", play=False, task_args={})
    scenes.apply(cfg, "seabed_rock")
    cfg.scene.num_envs = N
    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)
    env.reset()
    return env


@pytest.fixture(scope="module", params=["warp", "native"])
def env(request):
    e = _build(request.param)
    yield e
    e.close()


def _net_weight(env) -> np.ndarray:
    m0 = env.sim.mj_model
    tree = m0.body_rootid == m0.body_rootid[m0.body("robot/base_link").id]
    mass = env.sim.model.body_mass.cpu().numpy()
    gc = np.broadcast_to(np.asarray(env.sim.model.body_gravcomp.cpu().numpy()), mass.shape)
    return (mass[:, tree] * (1.0 - gc[:, tree])).sum(1) * G


def test_drawn_beta_is_the_simulated_beta(env):
    from oceanking.hydro import HydroCfg

    beta = env.ocean.beta.cpu().numpy()
    assert ((beta >= 0.75) & (beta <= 0.90)).all() and np.ptp(beta) > 0
    rho_v = 1000.0 * HydroCfg(buoyancy_ratio=0.8).ballast().total_volume
    np.testing.assert_allclose(_net_weight(env), (1 - beta) * rho_v / beta * G, rtol=1e-4)


def test_each_world_has_its_own_current(env):
    c = env.ocean.current_w.cpu().numpy()
    assert np.ptp(np.linalg.norm(c, axis=1)) > 0
    if hasattr(env.sim, "wp_model") and not type(env.sim).__module__.startswith("mjrl"):
        sim_c = env.sim.wp_model.opt.wind.numpy()
    else:
        sim_c = np.array([m.opt.wind for m in env.sim._models])
    np.testing.assert_allclose(sim_c, c, atol=1e-6)


def test_ground_friction_is_written_on_the_ground_that_outranks_the_feet(env):
    m0 = env.sim.mj_model
    g = m0.geom("terrain_0").id
    assert m0.geom_priority[g] > max(m0.geom_priority[i] for i in range(m0.ngeom)
                                     if m0.geom(i).name.startswith("robot/"))
    np.testing.assert_allclose(env.sim.model.geom_friction[:, g, 0].cpu().numpy(),
                               env.ocean.friction.cpu().numpy(), atol=1e-6)


def test_the_water_is_appended_after_upstreams_observation(env):
    obs = env.observation_manager.compute()
    assert obs["actor"].shape[1] == 415 + 4 and obs["critic"].shape[1] == 602 + 5
    assert env.observation_manager.active_terms["actor"][-2:] == ["buoyancy_ratio", "current_body"]
    torch.testing.assert_close(obs["critic"][:, 602], env.ocean.beta)


def test_the_current_is_observed_in_the_heading_frame(env):
    from oceanking import water

    q = env.scene["robot"].data.root_link_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    body = water.current_body(env)
    # Rotating the body-frame reading back by yaw recovers the world current.
    wx = torch.cos(yaw) * body[:, 0] - torch.sin(yaw) * body[:, 1]
    wy = torch.sin(yaw) * body[:, 0] + torch.cos(yaw) * body[:, 1]
    torch.testing.assert_close(torch.stack([wx, wy], 1), env.ocean.current_w[:, :2],
                               atol=1e-5, rtol=1e-5)


def test_the_mirror_negates_the_currents_left_component(env):
    from tasks.jumper.common.mdp.symmetry import _mirror_group

    obs = env.observation_manager.compute()["actor"]
    mirrored = _mirror_group(env, "actor", obs)
    torch.testing.assert_close(mirrored[:, -4], obs[:, -4])  # beta
    torch.testing.assert_close(mirrored[:, -3], obs[:, -3])  # forward
    torch.testing.assert_close(mirrored[:, -2], -obs[:, -2])  # left
    # Control: upstream's own terms do change under the mirror.
    assert not torch.allclose(mirrored[:, :415], obs[:, :415])


def test_the_current_curriculum_counts_its_own_steps(env):
    """Resuming restores mjlab's step counter to millions; the curriculum must not
    read it. Control: with the counter moved, the ceiling stays at the start."""
    import tasks.ocean.walk.env_cfg as walk

    env.common_step_counter = 10_000_000
    env.reset()
    assert env.ocean.current_max == pytest.approx(walk.CURRENT_START, abs=1e-3)
    assert (torch.linalg.norm(env.ocean.current_w, dim=1) <= walk.CURRENT_START + 1e-6).all()


def test_nominal_is_refused_in_training():
    with pytest.raises(ValueError, match="replays"):
        tasks.load_env_cfg("ocean.walk", play=False, task_args={"nominal": True})


def test_pinned_water_is_pinned():
    cfg = tasks.load_env_cfg("ocean.walk", play=True, task_args={
        "buoyancy_ratio": 0.85, "current": (-0.1, 0.05), "ground_friction": 0.5})
    assert cfg.events["ocean_buoyancy"].params["beta"] == 0.85
    assert cfg.events["ocean_water"].params["current"] == (-0.1, 0.05)
    assert cfg.events["ocean_water"].params["friction"] == 0.5
    assert math.isclose(cfg.events["ocean_water"].params["current_max"], 0.2)


def test_the_expanded_checkpoint_acts_exactly_as_the_official_one(tmp_path):
    from oceanking.tools.expand_checkpoint import OFFICIAL, expand

    expand(OFFICIAL, tmp_path)
    off = torch.load(OFFICIAL, map_location="cpu", weights_only=False)
    new = torch.load(tmp_path / OFFICIAL.name, map_location="cpu", weights_only=False)

    def act(sd, x):
        x = (x - sd["obs_normalizer._mean"]) / (sd["obs_normalizer._std"] + 1e-2)
        for i in (0, 2, 4, 6):
            x = torch.nn.functional.elu(x @ sd[f"mlp.{i}.weight"].T + sd[f"mlp.{i}.bias"])
        return x @ sd["mlp.8.weight"].T + sd["mlp.8.bias"]

    torch.manual_seed(0)
    x = torch.randn(32, 415)
    water = torch.randn(32, 4) * 3
    torch.testing.assert_close(act(new["actor_state_dict"], torch.cat([x, water], 1)),
                               act(off["actor_state_dict"], x), atol=0, rtol=0)
    assert new["iter"] == off["iter"] and new["infos"] == off["infos"]
    shapes = [s["exp_avg"].shape for s in new["optimizer_state_dict"]["state"].values()]
    assert (512, 419) in shapes and (512, 607) in shapes


# ── feasible-velocity target ────────────────────────────────────────────────


def test_a_feasible_command_is_its_own_target():
    from oceanking.water import project_to_disk

    cmd = torch.tensor([[0.1, 0.0], [0.0, -0.05]])
    target, inside = project_to_disk(cmd, torch.zeros(2, 2), torch.tensor([0.3, 0.3]))
    torch.testing.assert_close(target, cmd)
    assert inside.all()


def test_an_infeasible_command_becomes_the_most_the_physics_allows_towards_it():
    """Still water, limit 0.3: a 0.8 m/s command is tracked at 0.3 in its
    direction. Head current 0.2 (centre at -0.2), limit 0.3: forward at most 0.1."""
    from oceanking.water import project_to_disk

    target, inside = project_to_disk(torch.tensor([[0.8, 0.0], [0.8, 0.0]]),
                                     torch.tensor([[0.0, 0.0], [-0.2, 0.0]]),
                                     torch.tensor([0.3, 0.3]))
    torch.testing.assert_close(target, torch.tensor([[0.3, 0.0], [0.1, 0.0]]))
    assert not inside.any()


def test_standing_in_a_current_beyond_the_limit_is_scored_as_drifting():
    """A current the feet cannot hold against (0.4 against a limit of 0.3): the
    best possible is drifting downstream at 0.1, and that is the target."""
    from oceanking.water import project_to_disk

    target, inside = project_to_disk(torch.zeros(1, 2), torch.tensor([[0.0, -0.4]]),
                                     torch.tensor([0.3]))
    torch.testing.assert_close(target, torch.tensor([[0.0, -0.1]]))
    assert not inside.item()


def test_the_traction_limit_follows_the_physics(env):
    from oceanking import water

    limit = water.traction_limit(env).cpu().numpy()
    o = env.ocean
    expected = np.sqrt(2 * o.friction.cpu().numpy() * o.net_weight.cpu().numpy()
                       / (1000.0 * water.drag_area()))
    np.testing.assert_allclose(limit, expected, rtol=1e-5)
    np.testing.assert_allclose(o.net_weight.cpu().numpy(), _net_weight(env), rtol=1e-4)


def test_the_walk_task_tracks_the_feasible_velocity():
    from oceanking import water

    cfg = tasks.load_env_cfg("ocean.walk", play=False, task_args={})
    track = cfg.rewards["track_linear_velocity"]
    assert track.func is water.track_feasible_velocity
    # std is upstream's curriculum's; the rest are ocean.walk's.
    assert set(track.params) == {"std", "command_name", "margin", "std_fraction", "std_min",
                                 "target_fraction"}
    assert cfg.rewards["water_progress"].func is water.progress_towards_command
    assert isinstance(cfg.commands["twist"], water.OceanVelocityCommandCfg)


def test_trajectory_paths_keep_their_dots(tmp_path):
    """A stem like "b0.75_mu0.4" must not lose ".4" to the file extension."""
    from oceanking.trajectory import TrajectoryRecorder

    rec = TrajectoryRecorder.__new__(TrajectoryRecorder)
    rec._path, rec._meta, rec._step_dt = tmp_path / "b0.75_mu0.4", {}, 0.005
    rec._t, rec._pos = [0.0], [np.zeros((1, 3))]
    rec._quat, rec._cmd, rec._episode_step = [np.zeros((1, 4))], [np.zeros((1, 3))], [np.zeros(1)]
    rec.save()
    assert (tmp_path / "b0.75_mu0.4.npz").is_file() and (tmp_path / "b0.75_mu0.4.json").is_file()


def test_beyond_the_target_faster_towards_the_command_is_not_charged():
    """Limit 0.2 in still water, command 0.8 forward: walking at 0.2, 0.3 or 0.8
    costs nothing; 0.1 costs the shortfall; 0.9 the overshoot; drifting sideways
    costs its sideways part. Control: inside the disk it is upstream's error."""
    from oceanking.water import feasible_error_sq

    z = torch.zeros(1, 2)
    r = torch.tensor([0.2])
    cmd = torch.tensor([[0.8, 0.0]])
    def err(vx, vy=0.0):
        return feasible_error_sq(cmd, torch.tensor([[vx, vy]]), z, r).item()
    assert err(0.2) == err(0.3) == err(0.8) == 0.0
    assert err(0.1) == pytest.approx(0.01)
    assert err(0.9) == pytest.approx(0.01)
    assert err(0.3, 0.05) == pytest.approx(0.0025)
    inside = feasible_error_sq(torch.tensor([[0.1, 0.0]]), torch.tensor([[0.15, 0.0]]), z,
                               torch.tensor([0.2])).item()
    assert inside == pytest.approx(0.0025)


def test_in_a_head_current_the_disk_moves_downstream():
    """Head current 0.2 (centre at -0.2 in the body frame), limit 0.3: the most
    one can walk forward over the ground is 0.1; standing still is reachable."""
    from oceanking.water import feasible_error_sq

    centre, r = torch.tensor([[-0.2, 0.0]]), torch.tensor([0.3])
    fwd = torch.tensor([[0.8, 0.0]])
    assert feasible_error_sq(fwd, torch.tensor([[0.1, 0.0]]), centre, r).item() == 0.0
    assert feasible_error_sq(fwd, torch.tensor([[0.0, 0.0]]), centre, r).item() == pytest.approx(0.01)
    stand = torch.zeros(1, 2)
    assert feasible_error_sq(stand, torch.tensor([[-0.05, 0.0]]), centre, r).item() == pytest.approx(0.0025)


def test_progress_pays_for_speed_towards_an_out_of_reach_command():
    """Below the limit the progress term keeps rising with speed (no saturation),
    and it is zero for reachable commands."""
    from types import SimpleNamespace

    from oceanking import water

    class Env(SimpleNamespace):
        pass

    def env_with(cmd, vel):
        robot = SimpleNamespace(data=SimpleNamespace(
            root_link_lin_vel_b=torch.tensor([[vel[0], vel[1], 0.0]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]])))
        e = Env(num_envs=1, device="cpu", scene={"robot": robot},
                command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([cmd])))
        e.ocean = water.OceanState(beta=torch.tensor([0.8]), current_w=torch.zeros(1, 3),
                                   friction=torch.tensor([0.5]), load_scale=torch.ones(1),
                                   net_weight=torch.tensor([10.8]))
        return e

    far = [0.8, 0.0, 0.0]
    p = [water.progress_beyond_reach(env_with(far, (v, 0.0))).item() for v in (0.0, 0.1, 0.2, 0.3)]
    assert p[0] == 0.0 and p[0] < p[1] < p[2] < p[3]
    assert water.progress_beyond_reach(env_with([0.1, 0.0, 0.0], (0.1, 0.0))).item() == 0.0


def _point_env(cmd, vel):
    from types import SimpleNamespace

    from oceanking import water

    class Env(SimpleNamespace):
        pass

    robot = SimpleNamespace(data=SimpleNamespace(
        root_link_lin_vel_b=torch.tensor([[vel[0], vel[1], 0.0]]),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]])))
    e = Env(num_envs=1, device="cpu", scene={"robot": robot},
            command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([cmd])))
    e.ocean = water.OceanState(beta=torch.tensor([0.8]), current_w=torch.zeros(1, 3),
                               friction=torch.tensor([0.5]), load_scale=torch.ones(1),
                               net_weight=torch.tensor([10.8]))
    return e


def test_progress_pays_command_gain_for_reachable_commands():
    """Run 4: a reachable command of at least reachable_min pays its ground speed
    along it as a share of it, saturating at the command; tiny commands, none."""
    from oceanking import water

    kw = dict(std=0.2, std_fraction=0.4, std_min=0.06, target_fraction=0.5, reachable_min=0.08)
    small = [0.15, 0.0, 0.0]
    p = [water.progress_towards_command(_point_env(small, (v, 0.0)), **kw).item()
         for v in (0.0, 0.05, 0.10, 0.15, 0.20)]
    assert p[0] == 0.0 and p[1] < p[2] < p[3]
    assert p[3] == pytest.approx(1.0) and p[4] == pytest.approx(1.0)
    # Off its line it pays less; below reachable_min, nothing.
    assert water.progress_towards_command(_point_env(small, (0.15, 0.1)), **kw).item() < 0.5
    assert water.progress_towards_command(_point_env([0.05, 0.0, 0.0], (0.05, 0.0)), **kw).item() == 0.0
    # Out of reach: the same as progress_beyond_reach.
    far = [0.8, 0.0, 0.0]
    assert water.progress_towards_command(_point_env(far, (0.2, 0.0)), **kw).item() == pytest.approx(
        water.progress_beyond_reach(_point_env(far, (0.2, 0.0)), std=0.2, std_fraction=0.4, std_min=0.06))


def test_tracking_width_narrows_with_a_small_moving_target():
    """Run 4: the kernel is target_fraction x the target's speed for a moving
    target, the limit-based width for a standing one, never below std_min."""
    from oceanking import water

    env = _point_env([0.15, 0.0, 0.0], (0.0, 0.0))
    wide = water.tracking_std(env, 0.2, 0.4, 0.06)
    assert wide.item() > 0.1
    for speed, expect in ((0.15, 0.075), (0.08, 0.06), (0.0, wide.item()), (0.8, wide.item())):
        w = water.tracking_std(env, 0.2, 0.4, 0.06, torch.tensor([[speed, 0.0]]), 0.5)
        assert w.item() == pytest.approx(expect)
    # A 0.10 m/s walk for a 0.15 command now costs more than 0.5 of the term.
    r = water.track_feasible_velocity(_point_env([0.15, 0.0, 0.0], (0.10, 0.0)), std=0.2,
                                      command_name="twist", std_fraction=0.4, std_min=0.06,
                                      target_fraction=0.5)
    assert r.item() < 0.7


def test_straight_share_clears_yaw_and_heading(env):
    from oceanking import water

    term = env.command_manager.get_term("twist")
    saved = term.cfg.straight_share, term.cfg.reachable_share
    term.cfg.straight_share, term.cfg.reachable_share = 1.0, 1.0
    ids = torch.arange(env.num_envs, device=env.device)
    term._resample_command(ids)
    moving = ~term.is_standing_env
    assert torch.all(term.vel_command_b[moving, 2] == 0.0)
    assert not term.is_world_env[moving].any()
    if hasattr(term, "is_heading_env"):
        assert not term.is_heading_env[moving].any()
    term.cfg.straight_share, term.cfg.reachable_share = saved
    assert isinstance(term.cfg, water.OceanVelocityCommandCfg) and saved[0] == 0.5


def test_reachable_commands_are_drawn_inside_the_walkable_disk(env):
    """A share of moving commands is redrawn inside walk_fraction x the traction
    limit around the current; control: upstream's draw reaches far outside it."""
    from oceanking import water

    term = env.command_manager.get_term("twist")
    assert isinstance(term, water.OceanVelocityCommand)
    # The top curriculum rung's ranges, as a resumed run has them (this env starts
    # at rung 0, where upstream's own draw mostly lands inside the disk anyway).
    ranges = term.cfg.ranges
    saved = (ranges.lin_vel_x, ranges.lin_vel_y)
    ranges.lin_vel_x = ranges.lin_vel_y = (-0.8, 0.8)
    ids = torch.arange(env.num_envs, device=env.device)
    inside, outside = 0, 0
    for _ in range(40):
        term._resample_command(ids)
        cmd = term.vel_command_b[:, :2]
        centre = water.current_body(env)[:, :2]
        d = torch.linalg.norm(cmd - centre, dim=1)
        r = term.cfg.walk_fraction * water.traction_limit(env)
        moving = ~term.is_standing_env
        inside += int(((d <= r + 1e-5) & moving).sum())
        outside += int(((d > r + 1e-5) & moving).sum())
    ranges.lin_vel_x, ranges.lin_vel_y = saved
    share = inside / max(inside + outside, 1)
    # Half redrawn inside, plus the few upstream draws that land inside (~5%).
    assert 0.35 < share < 0.75
    assert outside > 0


def test_the_current_reading_has_noise_on_the_horizontal_only(env):
    from oceanking import water

    env.cfg.observations["actor"].enable_corruption = True
    try:
        noisy = water.current_body(env, noise=0.03)
    finally:
        env.cfg.observations["actor"].enable_corruption = False
    clean = water.current_body(env)
    assert torch.equal(noisy[:, 2], clean[:, 2])
    assert (noisy[:, :2] - clean[:, :2]).abs().max() <= 0.03 + 1e-6
    torch.testing.assert_close(water.current_body(env, noise=0.03), clean)  # corruption off


def test_the_expanded_checkpoint_asks_for_a_fresh_optimizer(tmp_path):
    from oceanking.tools.expand_checkpoint import OFFICIAL, expand
    from tasks.ocean.walk.rl_cfg import FRESH_OPTIMIZER_KEY, agent_cfg

    expand(OFFICIAL, tmp_path)
    assert torch.load(tmp_path / OFFICIAL.name, map_location="cpu",
                      weights_only=False)[FRESH_OPTIMIZER_KEY] is True
    cfg = agent_cfg()
    assert cfg.actor.distribution_cfg["learn_std"] is False
    assert cfg.algorithm.desired_kl == 0.005 and cfg.algorithm.learning_rate == 1e-4
