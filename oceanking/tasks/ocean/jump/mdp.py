"""ocean.jump's own terms: a jump aimed at a landing point (docs/JUMP.md).

The command is upstream's reference-free jump (`RefFreeJumpCommand`: one jump per
episode, whose flight, apex and landing it tracks) plus a **target**: a strength
s in [0, 1] and a direction, landing s x `reach` metres from the take-off point in
that direction (s = 0 is upstream's straight-up jump). The target is drawn once
per episode, in the world frame (every episode spawns at its origin facing +x),
and the actor sees it in its heading frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from oceanking import water

from ...jumper.ref_free_jump.mdp.commands import RefFreeJumpCommand, RefFreeJumpCommandCfg

#: Floor of the scaled contact threshold, N (a foot at beta 0.98 stands on ~0.15 N).
MIN_CONTACT_N = 0.02
#: Body vertical speed, m/s, above which a flight's foot contact is still the push.
RISING_SPEED = 0.05


@dataclass(kw_only=True)
class OceanJumpCommandCfg(RefFreeJumpCommandCfg):
    reach: float = 0.3
    """Landing distance at full strength, m (docs/JUMP.md section 2: what drag
    leaves of a realistic take-off)."""

    upright_share: float = 0.2
    """Share of episodes commanded straight up (s = 0)."""

    target: tuple[float, float] | None = None
    """A fixed (strength, direction in degrees) for every episode, for replays."""

    def __post_init__(self):
        self.class_type = OceanJumpCommand

    def build(self, env: Any) -> "OceanJumpCommand":
        return OceanJumpCommand(self, env)


class OceanJumpCommand(RefFreeJumpCommand):
    cfg: OceanJumpCommandCfg

    def __init__(self, cfg: OceanJumpCommandCfg, env: Any):
        super().__init__(cfg, env)
        n = self.num_envs
        #: Target landing point relative to the env origin, world frame, m.
        self.target = torch.zeros(n, 2, device=self.device)
        self.strength = torch.zeros(n, device=self.device)
        #: Where the episode's own jump landed (latched), and whether it has.
        self.landing = torch.zeros(n, 2, device=self.device)
        self._was_landed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._landing_error = torch.zeros(n, device=self.device)
        self._landing_distance = torch.zeros(n, device=self.device)
        self.metrics["landing_error"] = self._landing_error
        self.metrics["landing_distance"] = self._landing_distance
        self.metrics["strength"] = self.strength
        self._base_threshold = float(cfg.force_threshold)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        n = len(env_ids)
        if self.cfg.target is not None:
            s = torch.full((n,), float(self.cfg.target[0]), device=self.device)
            a = torch.full((n,), math.radians(self.cfg.target[1]), device=self.device)
        else:
            s = torch.rand(n, device=self.device)
            s[torch.rand(n, device=self.device) < self.cfg.upright_share] = 0.0
            a = 2.0 * math.pi * torch.rand(n, device=self.device)
        # A spawn past the push cannot make a jump of its own: nothing to aim.
        s = torch.where(self.can_jump[env_ids], s, torch.zeros_like(s))
        self.strength[env_ids] = s
        r = s * self.cfg.reach
        self.target[env_ids, 0] = r * torch.cos(a)
        self.target[env_ids, 1] = r * torch.sin(a)
        self.landing[env_ids] = 0.0
        self._was_landed[env_ids] = False

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        # A foot is "in contact" above upstream's 1 N on land, scaled per
        # environment by its net weight: at beta 0.9 a standing foot carries 0.8 N,
        # at 0.98 0.15 N, and with the land threshold the robot read as airborne
        # while standing -- every jump taken off at the spawn, none ever landed
        # (runs 1-3).
        scale = water.ocean_state(self._env).load_scale
        threshold = (self._base_threshold * scale).clamp_min(MIN_CONTACT_N)
        # A touch while the body still rises is part of the push, not a landing:
        # under water the official jump pushes twice (the middle legs touch again
        # 0.3 s in, body rising), and upstream's logic latched that as the landing
        # at the take-off point and the real flight as a second jump, clawing its
        # height back (probe of 2026-10-26).
        rising = self.robot.data.root_link_lin_vel_w[:, 2] > RISING_SPEED
        threshold = torch.where(self.in_flight & rising, torch.full_like(threshold, math.inf),
                                threshold)
        self.cfg.force_threshold = threshold.unsqueeze(1)
        super()._update_command(env_ids)
        if env_ids is not None:
            return
        new = self.landed & ~self._was_landed
        if new.any():
            self.landing[new] = self.position()[new]
        self._was_landed |= self.landed

    def _update_metrics(self) -> None:
        super()._update_metrics()
        err = torch.linalg.norm(self.landing - self.target, dim=1)
        self._landing_error[:] = torch.where(self.landed, err, torch.zeros_like(err))
        dist = torch.linalg.norm(self.landing, dim=1)
        self._landing_distance[:] = torch.where(self.landed, dist, torch.zeros_like(dist))
        self.metrics["landing_error"] = self._landing_error
        self.metrics["landing_distance"] = self._landing_distance
        self.metrics["strength"] = self.strength

    @property
    def jump_done(self) -> torch.Tensor:
        """Upstream's, but never while airborne on the episode's own jump: under
        water a flight outlasts the recording's span (airtime up to ~6 s at beta
        0.98), and upstream's `back_home` would end the episode mid-flight with
        the legs at HOME."""
        return self.landed | ((self.time_since_go > self.reference.span) & ~self.in_flight)

    def position(self) -> torch.Tensor:
        """Root position relative to the env origin, xy, m."""
        return (self.robot.data.root_link_pos_w[:, :2]
                - self._env.scene.env_origins[:, :2])


def _heading_frame(env: Any, v: torch.Tensor) -> torch.Tensor:
    q = env.scene["robot"].data.root_link_quat_w
    yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                      1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1],
                        torch.zeros_like(v[:, 0])], dim=1)


# ── observations ────────────────────────────────────────────────────────────


def jump_target(env: Any, command_name: str = "jump", mirror_kind: str = "vec3") -> torch.Tensor:
    """The landing target from the take-off point, in the heading frame, in units
    of `reach` (so its length is the strength): (forward, left, 0). Three
    components because upstream's mirror has a 3-vector kind and no 2-vector one."""
    cmd = env.command_manager.get_term(command_name)
    return _heading_frame(env, cmd.target / cmd.cfg.reach)


def target_offset(env: Any, command_name: str = "jump", mirror_kind: str = "vec3") -> torch.Tensor:
    """Critic only: the target from where the robot is now, heading frame, m."""
    cmd = env.command_manager.get_term(command_name)
    return _heading_frame(env, cmd.target - cmd.position())


# ── rewards ─────────────────────────────────────────────────────────────────


def landing_accuracy(cmd: OceanJumpCommand, sigma: float) -> torch.Tensor:
    """exp(-|landing - target|^2 / sigma^2), latched at the landing."""
    return torch.exp(-(cmd.landing - cmd.target).square().sum(dim=1) / sigma**2)


def landing_target(env: Any, command_name: str = "jump", sigma: float = 0.08) -> torch.Tensor:
    """Paid every step once the episode's own jump has landed (as upstream's
    `apex_height` is), nothing after a second jump."""
    cmd = env.command_manager.get_term(command_name)
    paying = (cmd.landed & ~cmd.second_jump).float()
    return landing_accuracy(cmd, sigma) * paying


def apex_on_target(env: Any, command_name: str = "jump", scale: float = 0.10,
                   sigma: float = 0.08) -> torch.Tensor:
    """Upstream's `apex_height` (its clawback on a second jump included), times
    the landing accuracy: height counts only on a jump that lands where it was
    aimed, so a high jump cannot be bought with the distance."""
    from ...jumper.ref_free_jump.mdp.rewards import apex_height

    cmd = env.command_manager.get_term(command_name)
    return apex_height(env, command_name, scale) * landing_accuracy(cmd, sigma)


def flight_progress(env: Any, command_name: str = "jump", max_speed: float = 1.0) -> torch.Tensor:
    """In the episode's own flight, the speed towards the target (m/s, clipped to
    +/- `max_speed`): the rate the distance to it shrinks. Summed over a flight it
    is the distance gained towards the target, so it is bounded by the target's
    distance (passing it turns the sign), and it pays nothing on the ground, where
    shuffling would earn it. A dense signal for aiming: run 2's landing kernels,
    paid seconds after the push, left the target input unused."""
    cmd = env.command_manager.get_term(command_name)
    offset = cmd.target - cmd.position()
    dist = offset.norm(dim=1)
    u = offset / dist.clamp_min(1e-6).unsqueeze(1)
    v = env.scene["robot"].data.root_link_lin_vel_w[:, :2]
    toward = (v * u).sum(dim=1).clamp(-max_speed, max_speed)
    # Aimed episodes only, and not within a centimetre of the target.
    aimed = (cmd.strength > 0) & (dist > 0.01)
    return toward * (cmd.in_flight & aimed).float()


def anchored_drift(env: Any, command_name: str = "jump", scale: float = 0.10,
                   deadband: float = 0.0) -> torch.Tensor:
    """Upstream's `horizontal_drift` with a moving anchor: the take-off point
    until the jump, nothing in flight, the landing point after it. So the robot
    neither shuffles towards the target nor slides on after landing. Before the
    jump only the distance beyond `deadband` counts: an aimed push leans the body
    over the feet (run 1: the full penalty there held every jump vertical)."""
    cmd = env.command_manager.get_term(command_name)
    p = cmd.position()
    anchor = torch.where(cmd.landed.unsqueeze(1), cmd.landing, torch.zeros_like(p))
    d = (p - anchor).norm(dim=1)
    d = torch.where(cmd.landed, d, (d - deadband).clamp(min=0.0))
    return d.square() / scale**2 * (~cmd.in_flight).float()


__all__ = [
    "OceanJumpCommand", "OceanJumpCommandCfg", "anchored_drift", "apex_on_target", "flight_progress",
    "jump_target", "landing_target", "target_offset",
]
