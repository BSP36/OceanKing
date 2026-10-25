"""The wet-rock scene: no robot can leave the ground within an episode.

The silent failure: the scene has no out-of-bounds termination (upstream drops
it), so a robot that walks off the edge falls forever and the episode continues
-- found in review with an 8 m tile and no border. Checked against the task's
own episode length, command ceiling and spawn range, so a change to any of them
is caught here.
"""

from __future__ import annotations

import math

import pytest

tasks = pytest.importorskip("tasks")
pytest.importorskip("mjlab")


def test_no_episode_can_reach_the_edge():
    import scenes.seabed_rock as rock

    cfg = tasks.load_env_cfg("ocean.posture", play=False, task_args={})
    r = cfg.commands["twist"].ranges
    v = math.hypot(max(map(abs, r.lin_vel_x)), max(map(abs, r.lin_vel_y)))
    spawn = cfg.events["reset_base"].params["pose_range"]
    start = math.hypot(max(map(abs, spawn["x"])), max(map(abs, spawn["y"])))
    reach = start + v * cfg.episode_length_s
    assert reach < rock.TILE / 2 + rock.BORDER


def test_the_ground_extends_where_promised():
    import numpy as np
    import scenes
    from mjlab.scene import Scene

    import scenes.seabed_rock as rock

    cfg = tasks.load_env_cfg("ocean.posture", play=True, task_args={})
    scenes.apply(cfg, "seabed_rock")
    m = Scene(cfg.scene, device="cpu").compile()
    ground = [g for g in range(m.ngeom) if m.geom(g).name.startswith("terrain")]
    lo = np.min([m.geom_aabb[g][:3] - m.geom_aabb[g][3:] + m.geom_pos[g] for g in ground], 0)
    hi = np.max([m.geom_aabb[g][:3] + m.geom_aabb[g][3:] + m.geom_pos[g] for g in ground], 0)
    assert min(-lo[0], -lo[1], hi[0], hi[1]) >= rock.TILE / 2 + rock.BORDER - 0.01
