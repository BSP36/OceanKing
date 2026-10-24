"""ocean.posture trains with `jumper.posture`'s hyper-parameters and runner."""

from __future__ import annotations

import dataclasses

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ...jumper.posture import rl_cfg as _posture
from ...jumper.posture.rl_cfg import play_status, runner_cls

__all__ = ["agent_cfg", "play_status", "runner_cls"]


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    # Logs are filed under the task id (jumper/tests/test_log_layout.py).
    return dataclasses.replace(_posture.agent_cfg(), experiment_name="ocean.posture")
