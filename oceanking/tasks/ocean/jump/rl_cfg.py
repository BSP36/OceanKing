"""ocean.jump: `jumper.ref_free_jump`'s network and PPO, tuned for fine-tuning.

As ocean.walk (docs/TRAINING.md section 6.4): a fresh Adam at a low initial
learning rate and the action std fixed at the official checkpoint's (0.13-0.33
per joint), with no entropy bonus. `desired_kl` stays at upstream's 0.01: an aimed
jump is a new skill, not a correction of the old one (docs/JUMP.md section 5).
"""

from __future__ import annotations

import dataclasses

from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg

from ...jumper.ref_free_jump import rl_cfg as _jump
from ..common import fresh_optimizer_runner

__all__ = ["agent_cfg", "runner_cls"]

LEARNING_RATE = 1.0e-4
ENTROPY_COEF = 0.0


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    # Logs are filed under the task id (jumper/tests/test_log_layout.py).
    cfg = dataclasses.replace(_jump.agent_cfg(), experiment_name="ocean.jump")
    cfg.algorithm.learning_rate = LEARNING_RATE
    cfg.algorithm.entropy_coef = ENTROPY_COEF
    cfg.actor.distribution_cfg = {**cfg.actor.distribution_cfg, "learn_std": False}
    return cfg


def runner_cls() -> type:
    return fresh_optimizer_runner(MjlabOnPolicyRunner)
