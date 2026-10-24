"""Every OceanKing task and scene is linked into upstream Jumper's tree.

A scene registered in `oceanking.plugin.SCENES` without its symlink fails only
when someone selects it, and a task directory without its import line simply
does not exist; both are caught here instead.
"""

from __future__ import annotations

from pathlib import Path

from oceanking.plugin import SCENES, SCENES_DIR, TASKS_DIR

REPO = Path(__file__).resolve().parents[1]
JUMPER = REPO / "jumper"


def test_the_task_tree_is_linked():
    link = JUMPER / "tasks" / "ocean"
    assert link.is_symlink() and link.resolve() == (TASKS_DIR / "ocean").resolve()


def test_every_task_directory_is_imported_upstream():
    text = (JUMPER / "tasks" / "__init__.py").read_text(encoding="utf-8")
    for task in (TASKS_DIR / "ocean").iterdir():
        if (task / "env_cfg.py").is_file():
            assert f"from .ocean import {task.name}" in text, task.name


def test_every_scene_is_registered_and_linked():
    on_disk = {p.stem for p in SCENES_DIR.glob("*.py")}
    assert on_disk == set(SCENES)
    for scene_id in SCENES:
        link = JUMPER / "scenes" / f"{scene_id}.py"
        assert link.is_symlink(), f"missing: ln -s ../../oceanking/scenes/{scene_id}.py {link}"
        assert link.resolve() == (SCENES_DIR / f"{scene_id}.py").resolve()
