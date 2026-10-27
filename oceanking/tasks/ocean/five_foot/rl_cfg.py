"""ocean.five_foot trains with `jumper.five_foot`'s hyper-parameters and runner."""

from __future__ import annotations

import dataclasses

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ...jumper.five_foot import rl_cfg as _base
from ...jumper.five_foot.rl_cfg import runner_cls

__all__ = ["agent_cfg", "runner_cls"]


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    # Logs are filed under the task id (jumper/tests/test_log_layout.py).
    return dataclasses.replace(_base.agent_cfg(), experiment_name="ocean.five_foot")
