"""ocean.jump's environment. Every number here is from `docs/JUMP.md`."""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from oceanking import water
from oceanking.buoyancy import check_model
from oceanking.hydro import HydroCfg, install

from ...jumper.common.assets import JUMPER_XML
from ...jumper.ref_free_jump import env_cfg as _base
from . import mdp
from ..common import add_trajectory, make_nominal

#: The model is built at this condition (keel sized and trimmed for beta 0.8, the
#: crown tank half full); per-environment beta changes the keel's mass only.
BASE = dict(mode="full", buoyancy_ratio=0.8, crown_air=0.5)
#: Training ranges (docs/JUMP.md section 3).
BETA_RANGE = (0.75, 0.98)
FRICTION_RANGE = (0.4, 0.8)
#: Landing distance at full strength, m, and the share of straight-up episodes.
REACH = 0.3
UPRIGHT_SHARE = 0.2
#: Landing accuracy: a narrow kernel for precision and a wide one that still pays
#: a jump that lands 0.3 m off (run 1: with the narrow one alone, exp(-14) for
#: a missed full-strength target, the policy learnt to land where it took off).
LANDING_SIGMA = 0.08
LANDING_WEIGHT = 6.0
LANDING_WIDE_SIGMA = 0.25
LANDING_WIDE_WEIGHT = 4.0
#: The apex term (upstream's 4) is scaled by the wide accuracy: run 1, scaled by
#: the narrow one, lowered its jumps while it could not yet aim.
APEX_SIGMA = LANDING_WIDE_SIGMA
#: Weight of the in-flight progress towards the target (per m/s; summed over a
#: flight, per metre gained): a full-strength jump earns about 3, as much as the
#: landing terms pay over an episode (run 2's Episode_Reward/landing_near: 1.2).
FLIGHT_PROGRESS_WEIGHT = 10.0
#: Body travel before take-off that the drift term does not charge, m.
DRIFT_DEADBAND = 0.05
#: Upstream's 2.5 s episode is shorter than a flight at high beta (up to ~6 s).
EPISODE_S = 8.0


def env_cfg(
    asset: Path | None = None,
    play: bool = False,
    buoyancy_ratio: float | None = None,
    ground_friction: float | None = None,
    jump: tuple[float, float] | None = None,
    nominal: bool = False,
    trajectory_out: str | None = None,
) -> ManagerBasedRlEnvCfg:
    cfg = _base.env_cfg(asset=asset, play=play)
    base = HydroCfg(**BASE)
    check_model(asset or JUMPER_XML)
    install(cfg, base)
    cfg.episode_length_s = EPISODE_S

    # ── water per environment (no current yet: docs/JUMP.md section 3) ──
    cfg.events["ocean_buoyancy"] = EventTermCfg(
        func=water.BuoyancyStartup, mode="startup",
        params={"hydro": base.as_dict(), "beta_range": BETA_RANGE, "beta": buoyancy_ratio},
    )
    cfg.events["ocean_clock"] = EventTermCfg(func=water.WaterClock, mode="step")
    cfg.events["ocean_water"] = EventTermCfg(
        func=water.WaterReset, mode="reset",
        params={"current_max": 0.0, "current_start": 0.0, "ramp_steps": 1,
                "friction_range": FRICTION_RANGE, "current": (0.0, 0.0),
                "friction": ground_friction},
    )

    # ── the command: upstream's jump, aimed ──
    old = cfg.commands["jump"]
    fields = {k: getattr(old, k) for k in old.__dataclass_fields__ if k != "class_type"}
    cfg.commands["jump"] = mdp.OceanJumpCommandCfg(
        **fields, reach=REACH, upright_share=UPRIGHT_SHARE,
        target=tuple(jump) if jump is not None else None)

    # ── observation: upstream's, then the target and the water (appended last, so
    # the official checkpoint's columns keep their places) ──
    vec3, scalar = {"mirror_kind": "vec3"}, {"mirror_kind": "scalar"}
    actor = cfg.observations["actor"].terms
    actor["jump_target"] = ObservationTermCfg(func=mdp.jump_target, params=vec3)
    actor["buoyancy_ratio"] = ObservationTermCfg(func=water.buoyancy_ratio, params=scalar)
    critic = cfg.observations["critic"].terms
    critic["jump_target"] = ObservationTermCfg(func=mdp.jump_target, params=vec3)
    critic["buoyancy_ratio"] = ObservationTermCfg(func=water.buoyancy_ratio, params=scalar)
    critic["ground_friction"] = ObservationTermCfg(func=water.ground_friction, params=scalar)
    critic["target_offset"] = ObservationTermCfg(func=mdp.target_offset, params=vec3)

    # ── reward: land where aimed; height only on target; no in-flight height ──
    if "horizontal_drift" not in cfg.rewards or "apex_height" not in cfg.rewards:
        raise RuntimeError("upstream's jump rewards changed")
    drift = cfg.rewards.pop("horizontal_drift")
    cfg.rewards["anchored_drift"] = RewardTermCfg(
        func=mdp.anchored_drift, weight=drift.weight,
        params={**drift.params, "deadband": DRIFT_DEADBAND})
    apex = cfg.rewards["apex_height"]
    apex.func = mdp.apex_on_target
    apex.params = {**apex.params, "sigma": APEX_SIGMA}
    cfg.rewards["landing_target"] = RewardTermCfg(
        func=mdp.landing_target, weight=LANDING_WEIGHT, params={"sigma": LANDING_SIGMA})
    cfg.rewards["flight_progress"] = RewardTermCfg(
        func=mdp.flight_progress, weight=FLIGHT_PROGRESS_WEIGHT)
    cfg.rewards["landing_near"] = RewardTermCfg(
        func=mdp.landing_target, weight=LANDING_WIDE_WEIGHT,
        params={"sigma": LANDING_WIDE_SIGMA})
    # The in-flight height pays every airborne step: at beta 0.98 a flight lasts
    # seconds and would pay more than any landing (docs/JUMP.md section 4).
    if "flight_height_weight" in cfg.curriculum:
        cfg.curriculum["flight_height_weight"].params["end_weight"] = 0.0
    cfg.rewards["flight_height"].weight = 0.0

    print(f"[oceanking] ocean.jump base {base.describe()}; beta "
          f"{buoyancy_ratio if buoyancy_ratio is not None else BETA_RANGE}, ground friction "
          f"{ground_friction if ground_friction is not None else FRICTION_RANGE}, target "
          f"{tuple(jump) if jump is not None else f'drawn, reach {REACH} m'}")
    if nominal:
        make_nominal(cfg, play)
    if trajectory_out:
        add_trajectory(cfg, play, trajectory_out, {
            "task": "ocean.jump", "hydro": base.as_dict(), "buoyancy_ratio": buoyancy_ratio,
            "ground_friction": ground_friction, "jump": jump})
    return cfg
