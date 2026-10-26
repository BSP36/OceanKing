"""Per-environment water for training: buoyancy, current and ground friction.

The specification is `docs/TRAINING.md` (sections 3, 4 and 6). This module is the
mechanism; every range and weight lives in the task's `env_cfg.py`.

The model is built once by `oceanking.hydro` at a base condition (beta 0.8: keel
sized and trimmed for it). Per environment, on top of that:

- **beta**, drawn once at startup: the keel's mass (and inertia, in proportion) is
  set so that the whole robot has the drawn beta. The keel stays where the base
  condition trimmed it. Startup only: a mass change needs `set_const`, too dear to
  run at every reset; 4096 environments cover the range between them.
- **current**, drawn at every reset: `opt.wind`, one entry per environment. MuJoCo
  Warp reads `opt.wind[world % len]`, so the array is replaced by a per-world one
  and the CUDA graph re-captured once; later writes go into it in place. The native
  backend writes each environment's model copy.
- **ground friction**, drawn at every reset, on the ground geoms (which outrank the
  feet on the seabed scenes, so their friction is the one the solver uses).

`WaterClock` counts environment steps for the current's curriculum. mjlab's own
`common_step_counter` cannot be used: resuming from the official checkpoint
restores it to millions of steps, which would open the curriculum at once.

All state lives on `env.ocean` (`OceanState`), read by the observation, reward and
metric functions below.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from oceanking.hydro import HydroCfg

G = 9.81
KEEL = "robot/oceanking_keel"
GROUND = re.compile(r"^terrain(_\d+)?$")


@dataclass
class OceanState:
    beta: torch.Tensor  # [N]
    current_w: torch.Tensor  # [N, 3] world frame, m/s
    friction: torch.Tensor  # [N]
    load_scale: torch.Tensor  # [N] underwater net weight / land weight
    net_weight: torch.Tensor  # [N] N, weight minus buoyancy
    steps: int = 0
    current_max: float = 0.0


def ocean_state(env: Any) -> OceanState:
    state = getattr(env, "ocean", None)
    if state is None:
        n, dev = env.num_envs, env.device
        state = OceanState(
            beta=torch.full((n,), float("nan"), device=dev),
            current_w=torch.zeros((n, 3), device=dev),
            friction=torch.full((n,), float("nan"), device=dev),
            load_scale=torch.ones((n,), device=dev),
            net_weight=torch.full((n,), float("nan"), device=dev),
        )
        env.ocean = state
    return state


def _uniform(lo: float, hi: float, n: int, device: Any) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=device)


def _is_native(sim: Any) -> bool:
    return type(sim).__module__.startswith("mjrl.backend")


# ── events ──────────────────────────────────────────────────────────────────


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
class BuoyancyStartup:
    """Draw beta per environment and size the keel for it (startup event)."""

    def __init__(self, cfg: Any, env: Any) -> None:
        params = cfg.params
        base = HydroCfg(**params["hydro"])
        ballast = base.ballast()
        self._rho = base.water_density
        self._volume = ballast.total_volume
        self._rest = ballast.total_mass - ballast.keel_mass  # robot + crown with its water
        self._land_weight = ballast.robot_mass * G
        m = env.sim.mj_model
        self._keel = m.body(KEEL).id
        self._keel_mass = float(m.body_mass[self._keel])
        self._keel_inertia = torch.as_tensor(m.body_inertia[self._keel], device=env.device)
        ocean_state(env)

    def __call__(self, env: Any, env_ids: Any, hydro: dict, beta_range: tuple[float, float],
                 beta: float | None = None) -> None:
        state = ocean_state(env)
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        n = len(env_ids)
        b = (torch.full((n,), float(beta), device=env.device) if beta is not None
             else _uniform(*beta_range, n, env.device))
        total = self._rho * self._volume / b
        keel = total - self._rest
        if (keel <= 0).any():
            raise ValueError(f"beta range {beta_range} needs a negative keel")
        model = env.sim.model
        model.body_mass[env_ids, self._keel] = keel.to(model.body_mass.dtype)
        inertia = self._keel_inertia[None, :] * (keel / self._keel_mass)[:, None]
        model.body_inertia[env_ids, self._keel] = inertia.to(model.body_inertia.dtype)
        state.beta[env_ids] = b
        state.net_weight[env_ids] = (1.0 - b) * total * G
        state.load_scale[env_ids] = state.net_weight[env_ids] / self._land_weight


class WaterClock:
    """Count environment steps (step event); see the module docstring."""

    def __init__(self, cfg: Any, env: Any) -> None:
        ocean_state(env)

    def __call__(self, env: Any, env_ids: Any) -> None:
        ocean_state(env).steps += 1


@requires_model_fields("geom_friction")
class WaterReset:
    """Draw the current and the ground's friction per environment (reset event)."""

    def __init__(self, cfg: Any, env: Any) -> None:
        ocean_state(env)
        m = env.sim.mj_model
        self._ground = [g for g in range(m.ngeom) if GROUND.match(m.geom(g).name or "")]
        if not self._ground:
            raise RuntimeError("no ground geom matches terrain(_<n>)")
        self._wind = None
        if not _is_native(env.sim):
            import warp as wp

            sim = env.sim
            with wp.ScopedDevice(sim.wp_device):
                sim.wp_model.opt.wind = wp.zeros(env.num_envs, dtype=wp.vec3)
                sim.create_graph()
            self._wind = wp.to_torch(sim.wp_model.opt.wind)

    def _write_current(self, env: Any, env_ids: torch.Tensor, current: torch.Tensor) -> None:
        if self._wind is not None:
            self._wind[env_ids] = current.to(self._wind.dtype)
            return
        sim = env.sim
        models = getattr(sim, "_models", None)
        if models is not None:
            for i, c in zip(env_ids.tolist(), current.tolist()):
                models[i].opt.wind[:] = c
        else:
            values = ocean_state(env).current_w
            if env.num_envs > 1 and not torch.allclose(values, values[:1]):
                raise RuntimeError("native backend without per-env models: one current only")
            sim.mj_model.opt.wind[:] = current[0].tolist()

    def __call__(self, env: Any, env_ids: Any, current_max: float, current_start: float,
                 ramp_steps: int, friction_range: tuple[float, float],
                 current: tuple[float, float] | None = None,
                 friction: float | None = None) -> None:
        state = ocean_state(env)
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        n = len(env_ids)
        if current is not None:
            c = torch.tensor([current[0], current[1], 0.0], device=env.device).repeat(n, 1)
            state.current_max = math.hypot(*current)
        else:
            share = min(1.0, state.steps / max(ramp_steps, 1))
            state.current_max = current_start + (current_max - current_start) * share
            speed = _uniform(0.0, state.current_max, n, env.device)
            angle = _uniform(0.0, 2.0 * math.pi, n, env.device)
            c = torch.stack([speed * torch.cos(angle), speed * torch.sin(angle),
                             torch.zeros_like(speed)], dim=1)
        state.current_w[env_ids] = c
        self._write_current(env, env_ids, c)

        mu = (torch.full((n,), float(friction), device=env.device) if friction is not None
              else _uniform(*friction_range, n, env.device))
        friction = env.sim.model.geom_friction
        for g in self._ground:
            friction[env_ids, g, 0] = mu.to(friction.dtype)
        state.friction[env_ids] = mu

        log = env.extras.setdefault("log", {})
        log["Ocean/current_max"] = state.current_max


# ── observations ────────────────────────────────────────────────────────────


# Each observation takes `mirror_kind`: upstream's symmetry augmentation reads it
# from the term's params (tasks/jumper/common/mdp/symmetry.py), and the observation
# manager passes params on to the function.


def buoyancy_ratio(env: Any, mirror_kind: str = "scalar") -> torch.Tensor:
    return ocean_state(env).beta.unsqueeze(1)


def current_body(env: Any, mirror_kind: str = "vec3", noise: float = 0.0,
                 group: str = "actor") -> torch.Tensor:
    """The water's velocity in the robot's heading (yaw) frame: (forward, left, up).

    Three components, the vertical always 0 here: upstream's mirror has a 3-vector
    kind (negate y) and no 2-vector one. `noise` (uniform, +/-) goes on the two
    horizontal components only, and only while `group`'s corruption is on: noise
    on the always-zero vertical one was a pure-noise input (review of run 2)."""
    q = env.scene["robot"].data.root_link_quat_w  # w, x, y, z
    yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                      1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    c = ocean_state(env).current_w
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    out = torch.stack([cos * c[:, 0] + sin * c[:, 1], -sin * c[:, 0] + cos * c[:, 1], c[:, 2]],
                      dim=1)
    if noise > 0.0 and getattr(env.cfg.observations.get(group), "enable_corruption", False):
        out[:, :2] += (2.0 * torch.rand_like(out[:, :2]) - 1.0) * noise
    return out


def ground_friction(env: Any, mirror_kind: str = "scalar") -> torch.Tensor:
    return ocean_state(env).friction.unsqueeze(1)


# ── what the physics allows ─────────────────────────────────────────────────


def drag_area() -> float:
    """Cd * A of the standing robot, head-on, m^2 (side-on is the same within 1%)."""
    from oceanking.drag import _table

    head = _table()["head_on_x"]
    return float(head["calibrated_Cd"] * head["silhouette_m2"])


def traction_limit(env: Any, margin: float = 1.0, rho: float = 1000.0) -> torch.Tensor:
    """Speed through the water at which drag equals the feet's friction, m/s, [N]:
    ``v = margin * sqrt(2 mu (1 - beta) M g / (rho Cd A))`` (docs/TRAINING.md section 1)."""
    state = ocean_state(env)
    return margin * torch.sqrt(2.0 * state.friction * state.net_weight / (rho * drag_area()))


def project_to_disk(command: torch.Tensor, centre: torch.Tensor,
                    radius: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The point of the disk (centre, radius) nearest each command, and whether
    the command was inside. All [N, 2] / [N] in one frame."""
    offset = command - centre
    dist = torch.linalg.norm(offset, dim=1)
    inside = dist <= radius
    scale = torch.where(inside, torch.ones_like(dist), radius / dist.clamp_min(1e-9))
    return centre + offset * scale.unsqueeze(1), inside


def feasible_velocity(env: Any, command_name: str = "twist",
                      margin: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """The ground velocity to track, body frame, [N, 2], and whether the command
    itself was feasible, [N].

    Through the water the feet hold up to the traction limit in any direction, so
    the reachable ground velocities are a disk centred on the current, radius the
    limit. A command inside it is the target; one outside is replaced by the
    disk's nearest point: as fast as the physics allows towards the command."""
    cmd = env.command_manager.get_command(command_name)[:, :2]
    centre = current_body(env)[:, :2]
    return project_to_disk(cmd, centre, traction_limit(env, margin))


# ── rewards and metrics ─────────────────────────────────────────────────────

#: Below this command speed, m/s, a command is "stand" (upstream's threshold).
STANDING_SPEED = 0.05


def feasible_error_sq(command: torch.Tensor, actual: torch.Tensor, centre: torch.Tensor,
                      radius: torch.Tensor) -> torch.Tensor:
    """Squared tracking error against what the physics allows, [N].

    A command inside the reachable disk is tracked as upstream does. For one
    outside it, the velocity relative to the water is split along and across the
    direction towards the command: across is error; along, falling short of the
    limit is error, and so is passing the command itself, but anything between
    is not. So a margin set conservatively (the limit is an estimate) never caps
    a policy that can do better."""
    offset = command - centre
    dist = torch.linalg.norm(offset, dim=1)
    inside = dist <= radius
    u = offset / dist.clamp_min(1e-9).unsqueeze(1)
    rel = actual - centre
    along = (rel * u).sum(dim=1)
    across = rel - along.unsqueeze(1) * u
    outside = (across.square().sum(dim=1) + torch.relu(radius - along).square()
               + torch.relu(along - dist).square())
    within = (command - actual).square().sum(dim=1)
    return torch.where(inside, within, outside)


def tracking_std(env: Any, std: float, std_fraction: float | None, std_min: float,
                 target: torch.Tensor | None = None,
                 target_fraction: float | None = None) -> torch.Tensor:
    """The tracking kernel's width, [N]: upstream's land `std` (0.2 m/s), or, with
    `std_fraction`, that fraction of each environment's traction limit (clamped to
    [std_min, std]) -- so a shortfall of a given share of what is possible costs the
    same everywhere, instead of almost nothing where little is possible.

    With `target_fraction` and the target velocity ([N, 2]), a moving target
    narrows it further to that fraction of the target's speed (never below
    `std_min`): at the limit-based width alone (0.145 m/s at beta 0.8, friction
    0.5) walking 0.10 m/s for a 0.15 command cost only 0.22/s, less than the style
    terms pay for the slower gait (review of run 3). A target under the standing
    threshold keeps the limit-based width."""
    if std_fraction is None:
        return torch.full((env.num_envs,), float(std), device=env.device)
    width = (std_fraction * traction_limit(env)).clamp(std_min, std)
    if target_fraction is None or target is None:
        return width
    speed = torch.linalg.norm(target, dim=1)
    narrow = torch.minimum(width, (target_fraction * speed).clamp_min(std_min))
    return torch.where(speed >= STANDING_SPEED, narrow, width)


def track_feasible_velocity(env: Any, std: float, command_name: str, margin: float = 1.0,
                            std_fraction: float | None = None, std_min: float = 0.05,
                            target_fraction: float | None = None) -> torch.Tensor:
    """Upstream's `track_linear_velocity`, scored against what the physics allows
    (`feasible_error_sq`): full marks for the best possible (docs/TRAINING.md
    section 6). `std` is upstream's, which its curriculum sets; see `tracking_std`."""
    cmd = env.command_manager.get_command(command_name)[:, :2]
    actual = env.scene["robot"].data.root_link_lin_vel_b
    centre = current_body(env)[:, :2]
    limit = traction_limit(env, margin)
    error = feasible_error_sq(cmd, actual[:, :2], centre, limit)
    target = project_to_disk(cmd, centre, limit)[0] if target_fraction is not None else None
    width = tracking_std(env, std, std_fraction, std_min, target, target_fraction)
    return torch.exp(-(error + actual[:, 2].square()) / width.square())


def progress_beyond_reach(env: Any, command_name: str = "twist", std: float = 0.2,
                          std_fraction: float | None = None, std_min: float = 0.05) -> torch.Tensor:
    """For a command outside the reachable disk: the speed through the water towards
    it, as a share of the traction limit (clipped to [0, 1]), times staying on its
    line. Zero for reachable commands. A slope that does not saturate below the
    limit, so walking faster towards an out-of-reach command keeps paying (review
    of run 2: the exp kernel alone lost to the style terms)."""
    cmd = env.command_manager.get_command(command_name)[:, :2]
    actual = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    centre = current_body(env)[:, :2]
    limit = traction_limit(env)
    offset = cmd - centre
    dist = torch.linalg.norm(offset, dim=1)
    outside = dist > limit
    u = offset / dist.clamp_min(1e-9).unsqueeze(1)
    rel = actual - centre
    along = (rel * u).sum(dim=1)
    across = (rel - along.unsqueeze(1) * u).square().sum(dim=1)
    width = tracking_std(env, std, std_fraction, std_min)
    share = (along / limit.clamp_min(1e-6)).clamp(0.0, 1.0)
    return outside.float() * share * torch.exp(-across / width.square())


def progress_towards_command(env: Any, command_name: str = "twist", std: float = 0.2,
                             std_fraction: float | None = None, std_min: float = 0.05,
                             target_fraction: float | None = None,
                             reachable_min: float = 0.08) -> torch.Tensor:
    """`progress_beyond_reach`, extended to reachable commands of at least
    `reachable_min` m/s: there, the ground speed along the command as a share of
    it (clipped to [0, 1]), times staying on its line. A slope of 1/|command| that
    the exp kernel's flat top lacks, so a small command is worth following at its
    full speed (review of run 3: the policy's command gain kept shrinking because
    the style terms paid more for a slower gait than tracking charged)."""
    cmd = env.command_manager.get_command(command_name)[:, :2]
    actual = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    centre = current_body(env)[:, :2]
    limit = traction_limit(env)
    speed = torch.linalg.norm(cmd, dim=1)
    inside = (torch.linalg.norm(cmd - centre, dim=1) <= limit) & (speed >= reachable_min)
    u = cmd / speed.clamp_min(1e-9).unsqueeze(1)
    along = (actual * u).sum(dim=1)
    across = (actual - along.unsqueeze(1) * u).square().sum(dim=1)
    width = tracking_std(env, std, std_fraction, std_min, cmd, target_fraction)
    share = (along / speed.clamp_min(1e-6)).clamp(0.0, 1.0)
    within = inside.float() * share * torch.exp(-across / width.square())
    beyond = progress_beyond_reach(env, command_name, std, std_fraction, std_min)
    return within + beyond


def feasible_tracking_error(env: Any, command_name: str = "twist",
                            margin: float = 1.0) -> torch.Tensor:
    """sqrt of `feasible_error_sq`, m/s: what the tracking reward is charging."""
    cmd = env.command_manager.get_command(command_name)[:, :2]
    actual = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    return feasible_error_sq(cmd, actual, current_body(env)[:, :2],
                             traction_limit(env, margin)).sqrt()


def command_feasible(env: Any, command_name: str = "twist", margin: float = 1.0) -> torch.Tensor:
    return feasible_velocity(env, command_name, margin)[1].float()


def scaled_load(func: Any, base_threshold: float):
    """Wrap an upstream foot-load reward so its threshold follows the net weight."""

    def reward(env: Any, **params: Any) -> torch.Tensor:
        threshold = base_threshold * ocean_state(env).load_scale.unsqueeze(1)
        return func(env, force_threshold=threshold, **params)

    reward.__name__ = f"scaled_{getattr(func, '__name__', 'load')}"
    return reward


def standing_drift(env: Any, command_name: str = "twist", threshold: float = 0.05) -> torch.Tensor:
    """Ground speed while commanded to stand (m/s), zero while moving."""
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.linalg.norm(cmd[:, :3], dim=1) < threshold).float()
    v = env.scene["robot"].data.root_link_lin_vel_w[:, :2]
    return torch.linalg.norm(v, dim=1) * standing


def standing_share(env: Any, command_name: str = "twist", threshold: float = 0.05) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.linalg.norm(cmd[:, :3], dim=1) < threshold).float()


__all__ = [
    "BuoyancyStartup", "OceanState", "WaterClock", "WaterReset", "buoyancy_ratio", "current_body", "ground_friction", "ocean_state", "scaled_load",
    "standing_drift", "standing_share",
]


# ── commands ────────────────────────────────────────────────────────────────

from dataclasses import dataclass as _dataclass  # noqa: E402

from mjlab.tasks.velocity.mdp.velocity_command import (  # noqa: E402
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)


class OceanVelocityCommand(UniformVelocityCommand):
    """Upstream's velocity command, with a share of moving commands drawn inside
    what this environment can walk.

    Upstream draws each axis uniformly in +/-0.8 m/s; under water that put about
    six moving commands in seven out of reach, the size of a command stopped
    carrying any reward signal, and the policy's command gain shrank by a third,
    even for 0.15 m/s commands that are reachable everywhere (review of run 2).
    Here a share `reachable_share` of the commands upstream did not make standing
    is replaced by a point drawn uniformly in the disk the robot can walk: centred
    on the current (heading frame), radius `walk_fraction` of the traction limit
    (walking reaches about two thirds of the static limit, docs/TRAINING.md
    section 1). The rest keep upstream's draw, mostly out of reach: "as fast as
    possible that way"."""

    cfg: "OceanVelocityCommandCfg"

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        env = self._env
        state = getattr(env, "ocean", None)
        if state is None or len(env_ids) == 0:
            return
        limit = traction_limit(env)[env_ids]
        if not torch.isfinite(limit).all():
            return  # water not drawn yet (first reset): keep upstream's draw
        moving = ~self.is_standing_env[env_ids]
        pick = moving & (torch.rand(len(env_ids), device=self.device) < self.cfg.reachable_share)
        ids = env_ids[pick]
        if len(ids) == 0:
            return
        radius = self.cfg.walk_fraction * limit[pick]
        r = radius * torch.sqrt(torch.rand(len(ids), device=self.device))
        a = 2.0 * math.pi * torch.rand(len(ids), device=self.device)
        centre = current_body(env)[ids, :2]
        self.vel_command_b[ids, 0] = centre[:, 0] + r * torch.cos(a)
        self.vel_command_b[ids, 1] = centre[:, 1] + r * torch.sin(a)
        # The disk is drawn in the heading frame, so these commands stay body-frame
        # (a world-frame env would have rotated them away from the disk).
        self.is_world_env[ids] = False
        self.vel_command_w[ids] = self.vel_command_b[ids]
        # A share walk straight: upstream's yaw rate (uniform +/-4 rad/s, or heading
        # control) turned faster than 1 rad/s in 55% of small commands, so only
        # ~5% of training resembled walking straight at 0.1-0.2 m/s (review of run 3).
        straight = ids[torch.rand(len(ids), device=self.device) < self.cfg.straight_share]
        self.vel_command_b[straight, 2] = 0.0
        self.vel_command_w[straight, 2] = 0.0
        if hasattr(self, "is_heading_env"):
            self.is_heading_env[straight] = False


@_dataclass(kw_only=True)
class OceanVelocityCommandCfg(UniformVelocityCommandCfg):
    reachable_share: float = 0.5
    walk_fraction: float = 0.65
    #: Share of the reachable commands given no yaw rate (and no heading control).
    straight_share: float = 0.0

    def build(self, env: Any) -> OceanVelocityCommand:
        return OceanVelocityCommand(self, env)
