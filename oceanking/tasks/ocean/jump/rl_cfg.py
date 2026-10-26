"""ocean.jump trains with `jumper.ref_free_jump`'s hyper-parameters and runner."""

from __future__ import annotations

import dataclasses

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ...jumper.ref_free_jump import rl_cfg as _jump

__all__ = ["agent_cfg"] + [n for n in ("play_status", "runner_cls") if hasattr(_jump, n)]
globals().update({n: getattr(_jump, n) for n in __all__ if n != "agent_cfg"})


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    # Logs are filed under the task id (jumper/tests/test_log_layout.py).
    return dataclasses.replace(_jump.agent_cfg(), experiment_name="ocean.jump")
