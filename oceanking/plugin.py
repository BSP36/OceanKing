"""How OceanKing's tasks and scenes reach upstream Jumper's registries.

Upstream resolves a task id to the module `tasks.<id>` and a scene id to
`scenes.<id>`, and its test suite checks that each lives on disk under
`jumper/tasks/` and `jumper/scenes/`. So the sources live here and `jumper/`
holds **symlinks** to them, plus one marked (`[oceanking]`) line in each
package's `__init__.py`:

    jumper/tasks/ocean          -> oceanking/tasks/ocean
    jumper/scenes/<id>.py       -> oceanking/scenes/<id>.py

`oceanking/tasks/` and `oceanking/scenes/` have no `__init__.py`: their modules are
imported only as `tasks.ocean.*` and `scenes.<id>`, never as `oceanking.*`, which
would create a second copy of each.

To add a scene: write `oceanking/scenes/<id>.py`, add it to `SCENES` below, and
`ln -s ../../oceanking/scenes/<id>.py jumper/scenes/<id>.py`. To add a task: create
`oceanking/tasks/ocean/<name>/` and add one import line in `jumper/tasks/__init__.py`.
`tests/test_layout.py` checks that every one of them is linked.

Like upstream's registration, this must stay free of simulation dependencies:
`train.py --list` imports it on a machine without mjlab.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS_DIR = ROOT / "tasks"
SCENES_DIR = ROOT / "scenes"

#: Scene id -> (description, tags, use); see `scenes.registry.register`.
SCENES: dict[str, tuple[str, tuple[str, ...], str]] = {
    "seabed": (
        "underwater look: flat hard seabed (reference ground), water light fading with --depth",
        ("ocean", "underwater"),
        "both",
    ),
    "seabed_rock": (
        "**changes the ground**: wet rock, uneven (20 mm relief), friction 0.5; underwater look",
        ("ocean", "underwater", "terrain"),
        "both",
    ),
    "seabed_sand": (
        "**changes the ground**: seabed sand, feet sink, granular drag, friction 0.6; "
        "underwater look",
        ("ocean", "underwater"),
        "both",
    ),
}


def register_scenes() -> None:
    """Called from `jumper/scenes/__init__.py`."""
    from scenes.registry import register

    for scene_id, (description, tags, use) in SCENES.items():
        register(scene_id, description, tags, use=use)
