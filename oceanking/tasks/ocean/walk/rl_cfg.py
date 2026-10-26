"""ocean.walk: `jumper.posture`'s network and PPO, tuned for fine-tuning.

Changes from upstream, each from the review of runs 1-2 (docs/TRAINING.md
section 6.4):

- `desired_kl` 0.01 -> 0.005: at 0.01 the actor's weights random-walked about
  20% away from the official ones with no consistent direction, eroding the land
  gait where the underwater reward signal is weak.
- the action std is **fixed** at the official checkpoint's values (0.18-0.24 per
  joint, loaded from it); it had risen 0.20 -> 0.26, which upstream's own notes
  identify as a tax through `action_rate_l2`. The entropy bonus is then moot (0).
- the initial learning rate 1e-3 -> 1e-4, because the optimiser starts fresh
  (`OceanRunner`): the adaptive schedule then moves it, but not before the first
  update.
"""

from __future__ import annotations

import dataclasses

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ...jumper.posture import rl_cfg as _posture
from ...jumper.posture.rl_cfg import play_status
from ..common import FRESH_OPTIMIZER_KEY, fresh_optimizer_runner  # noqa: F401

__all__ = ["agent_cfg", "play_status", "runner_cls"]

DESIRED_KL = 0.005
LEARNING_RATE = 1.0e-4
ENTROPY_COEF = 0.0


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = _posture.agent_cfg()
    # Logs are filed under the task id (jumper/tests/test_log_layout.py).
    cfg = dataclasses.replace(cfg, experiment_name="ocean.walk")
    cfg.algorithm.desired_kl = DESIRED_KL
    cfg.algorithm.learning_rate = LEARNING_RATE
    cfg.algorithm.entropy_coef = ENTROPY_COEF
    cfg.actor.distribution_cfg = {**cfg.actor.distribution_cfg, "learn_std": False}
    return cfg


def runner_cls() -> type:
    return fresh_optimizer_runner(_posture.runner_cls())
