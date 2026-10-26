"""ocean.walk's environment. Every number here is from `docs/TRAINING.md`."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from oceanking import water
from oceanking.buoyancy import check_model
from oceanking.hydro import HydroCfg, install

from ...jumper.common.assets import JUMPER_XML
from ...jumper.posture.env_cfg import env_cfg as posture_env_cfg
from ...jumper.posture.rl_cfg import NUM_STEPS_PER_ENV
from ..common import add_trajectory, make_nominal

#: The model is built at this condition: keel sized and trimmed for beta 0.8 with
#: the crown tank half full. Per-environment beta then changes the keel's mass only.
BASE = dict(mode="full", buoyancy_ratio=0.8, crown_air=0.5)
#: Training ranges (TRAINING.md section 4): standing stays inside the traction limit
#: everywhere in them (worst corner, beta 0.9 on friction 0.4: 0.22 m/s).
BETA_RANGE = (0.75, 0.90)
CURRENT_MAX = 0.2  # m/s
CURRENT_START = 0.05  # m/s, the curriculum's first ceiling
CURRENT_RAMP_ITERATIONS = 1000
FRICTION_RANGE = (0.4, 0.8)
#: Upstream's base COM offset ranges are scaled by this: they were sized against
#: the land weight, about twice the underwater net weight.
BASE_COM_SCALE = 0.5
#: Flow-sensor noise on the observed current, m/s (uniform, +/-).
CURRENT_NOISE = 0.03
#: Multiplier on the static traction limit in the feasible-velocity target: 1, the
#: physical limit, as an aspiration. 0.65 (what the official gait reaches) was
#: tried and rejected before training: it made the official speed the ceiling of
#: what is rewarded (review of run 2).
TRACTION_MARGIN = 1.0
#: Walking reaches about two thirds of the static limit: the official policy's top
#: speed, commanded 0.8 m/s in still water on rock over beta 0.75-0.9 x friction
#: 0.4-0.8, was 0.67 (0.56-0.75) of it forward and 0.63 (0.39-0.73) sideways,
#: correlating 0.99 (measured 2026-10-25). Reachable commands are drawn inside it.
WALK_FRACTION = 0.65
#: Share of moving commands drawn inside the walkable disk (the rest: upstream's).
REACHABLE_SHARE = 0.5
#: Tracking kernel width as a share of the traction limit (upstream's 0.2 m/s land
#: width is the upper clamp): a 0.07 m/s shortfall of a 0.24 m/s target cost 0.2/s.
TRACK_STD_FRACTION = 0.4
#: Run 4: a moving target narrows the width further to this share of its speed
#: (review of run 3: at 0.4 v_lim, ~0.15 m/s, walking 0.10 for a 0.15 command cost
#: less than the style terms paid for the slower gait). Not below TRACK_STD_MIN:
#: the error includes the vertical speed.
TRACK_TARGET_FRACTION = 0.5
TRACK_STD_MIN = 0.06
#: Weight of the progress term for out-of-reach commands (slope ~4/(m/s) at
#: typical limits, against ~3/(m/s) the style terms pay for slowing; review of run 2).
#: Run 4: 1.5 -> 2.0, since it now also pays reachable commands, where 1.5 only broke
#: even with what the style terms pay for a shorter stride (review of run 3).
PROGRESS_WEIGHT = 2.0
#: Run 4: the progress term also pays reachable commands of at least this speed,
#: m/s, their ground speed along the command as a share of it.
PROGRESS_REACHABLE_MIN = 0.08
#: Run 4: share of the reachable commands that walk straight (no yaw rate).
STRAIGHT_SHARE = 0.5
#: Upstream's foot-load thresholds, N, scaled per environment by its net weight.
STANCE_LOAD_N = 5.0
STANCE_LOAD_STAND_N = 3.0


def env_cfg(
    asset: Path | None = None,
    play: bool = False,
    buoyancy_ratio: float | None = None,
    current: tuple[float, float] | None = None,
    ground_friction: float | None = None,
    nominal: bool = False,
    trajectory_out: str | None = None,
) -> ManagerBasedRlEnvCfg:
    cfg = posture_env_cfg(asset=asset, play=play)
    base = HydroCfg(**BASE)
    check_model(asset or JUMPER_XML)
    install(cfg, base)

    # ── water per environment ──
    cfg.events["ocean_buoyancy"] = EventTermCfg(
        func=water.BuoyancyStartup, mode="startup",
        params={"hydro": base.as_dict(), "beta_range": BETA_RANGE, "beta": buoyancy_ratio},
    )
    cfg.events["ocean_clock"] = EventTermCfg(func=water.WaterClock, mode="step")
    cfg.events["ocean_water"] = EventTermCfg(
        func=water.WaterReset, mode="reset",
        params={
            "current_max": CURRENT_MAX, "current_start": CURRENT_START,
            "ramp_steps": CURRENT_RAMP_ITERATIONS * NUM_STEPS_PER_ENV,
            "friction_range": FRICTION_RANGE,
            "current": tuple(current) if current is not None else None,
            "friction": ground_friction,
        },
    )
    if "base_com" in cfg.events:
        ranges = cfg.events["base_com"].params["ranges"]
        for axis, (lo, hi) in list(ranges.items()):
            ranges[axis] = (lo * BASE_COM_SCALE, hi * BASE_COM_SCALE)

    # ── observation: upstream's terms, then the water's (appended last, so the
    # official checkpoint's columns keep their places) ──
    actor = cfg.observations["actor"].terms
    # `mirror_kind` tells upstream's symmetry augmentation how each term reflects.
    scalar, vec3 = {"mirror_kind": "scalar"}, {"mirror_kind": "vec3"}
    actor["buoyancy_ratio"] = ObservationTermCfg(func=water.buoyancy_ratio, params=scalar)
    # Noise on the horizontal components only (inside the function): the manager's
    # noise also hit the always-zero vertical one, a pure-noise input.
    actor["current_body"] = ObservationTermCfg(
        func=water.current_body, params={**vec3, "noise": CURRENT_NOISE, "group": "actor"})
    critic = cfg.observations["critic"].terms
    critic["buoyancy_ratio"] = ObservationTermCfg(func=water.buoyancy_ratio, params=scalar)
    critic["current_body"] = ObservationTermCfg(func=water.current_body, params=vec3)
    critic["ground_friction"] = ObservationTermCfg(func=water.ground_friction, params=scalar)

    # ── reward: foot loads follow the net weight ──
    for name, newtons in (("stance_load", STANCE_LOAD_N), ("stance_load_stand", STANCE_LOAD_STAND_N)):
        term = cfg.rewards[name]
        if term.params.get("force_threshold") != newtons:
            raise RuntimeError(f"upstream's {name} threshold changed: "
                               f"{term.params.get('force_threshold')} N, expected {newtons}")
        term.func = water.scaled_load(term.func, newtons)
        term.params = {k: v for k, v in term.params.items() if k != "force_threshold"}

    # ── reward: track what the physics allows, not the raw command ──
    track = cfg.rewards["track_linear_velocity"]
    if set(track.params) != {"std", "command_name"}:
        raise RuntimeError(f"upstream's track_linear_velocity params changed: {track.params}")
    track.func = water.track_feasible_velocity
    track.params.update(margin=TRACTION_MARGIN, std_fraction=TRACK_STD_FRACTION,
                        std_min=TRACK_STD_MIN, target_fraction=TRACK_TARGET_FRACTION)
    cfg.rewards["water_progress"] = RewardTermCfg(
        func=water.progress_towards_command, weight=PROGRESS_WEIGHT,
        params={"command_name": "twist", "std": track.params["std"],
                "std_fraction": TRACK_STD_FRACTION, "std_min": TRACK_STD_MIN,
                "target_fraction": TRACK_TARGET_FRACTION,
                "reachable_min": PROGRESS_REACHABLE_MIN})

    # ── commands (training): a share drawn inside what this environment can walk.
    # A replay keeps upstream's operator command, which the pad, the keyboard and
    # `--command-script` drive. ──
    twist = cfg.commands["twist"]
    if not play:
        if type(twist) is not water.UniformVelocityCommandCfg:
            raise RuntimeError(f"upstream's twist command is a {type(twist).__name__} now")
        cfg.commands["twist"] = water.OceanVelocityCommandCfg(
            **{f.name: getattr(twist, f.name) for f in dataclasses.fields(twist)},
            reachable_share=REACHABLE_SHARE, walk_fraction=WALK_FRACTION,
            straight_share=STRAIGHT_SHARE)

    cfg.metrics["feasible_tracking_error"] = MetricsTermCfg(
        func=water.feasible_tracking_error, params={"margin": TRACTION_MARGIN})
    cfg.metrics["command_feasible"] = MetricsTermCfg(
        func=water.command_feasible, params={"margin": TRACTION_MARGIN})
    cfg.metrics["standing_drift"] = MetricsTermCfg(func=water.standing_drift)
    cfg.metrics["standing_share"] = MetricsTermCfg(func=water.standing_share)

    print(f"[oceanking] ocean.walk base {base.describe()}; beta "
          f"{buoyancy_ratio if buoyancy_ratio is not None else BETA_RANGE}, current "
          f"{tuple(current) if current is not None else f'0-{CURRENT_MAX} m/s'}, ground friction "
          f"{ground_friction if ground_friction is not None else FRICTION_RANGE}")
    if nominal:
        make_nominal(cfg, play)
    if trajectory_out:
        add_trajectory(cfg, play, trajectory_out, {
            "task": "ocean.walk", "hydro": base.as_dict(), "buoyancy_ratio": buoyancy_ratio,
            "current": current, "ground_friction": ground_friction})
    return cfg
