"""seabed_sand -- the sandy seabed: feet sink, sand resists, and grips at 0.6.

**This scene changes the physics** (ground contact and a per-step force), like
upstream's `soft`; the water itself still comes from the task.

## What is borrowed and what is not

The compliance (`solimp`, `solref`) and the granular drag on a sunk foot are
upstream's `scenes/soft.py`, unchanged: feet sink about 1 cm under the land
robot's weight and a buried foot is resisted sideways in proportion to its depth.
Upstream records that `DRAG_K` has no measured basis; that stays true here.

## Friction: 0.6

A foot sliding over saturated sand shears the sand itself, so the friction is
set by the sand's angle of internal friction, about 30-35 degrees for clean
sand: tan(30..35 deg) = 0.58-0.70. Taken at **0.6** (docs/PHYSICS.md). The ground
outranks the feet's contact priority, so this replaces the feet's dry 1.2.

Underwater the robot presses on the sand with its net weight, (1 - beta) of its
weight, so it sinks less than upstream measured on land.
"""

from __future__ import annotations

from scenes.registry import GROUND, Scene, set_ground_contact

from .seabed import DEFAULT_DEPTH_M, add_landmarks, underwater_light
from .seabed import cli_args as _depth_args

#: Sliding, torsional, rolling. Torsional and rolling are the feet's own.
SAND_FRICTION = (0.6, 0.01, 0.01)

SAND = (0.93, 0.80, 0.52)
SAND_DARK = (0.84, 0.71, 0.45)
SAND_MARK = (0.70, 0.58, 0.36)


def cli_args(group):
    return _depth_args(group)


def _decorate(spec) -> None:
    from scenes.soft import SOFT_SOLIMP, SOFT_SOLREF

    set_ground_contact(spec, "seabed_sand", SAND_FRICTION, SOFT_SOLREF, SOFT_SOLIMP)
    add_landmarks(spec)


def scene(depth: float = DEFAULT_DEPTH_M) -> Scene:
    from mjlab.utils import spec_config as sc
    from scenes.soft import _drag_event

    lights, headlight, sky = underwater_light(depth)
    return Scene(
        headlight=headlight,
        textures=(
            sc.TextureCfg(
                name="seabed_sand", type="2d", builtin="checker",
                rgb1=SAND, rgb2=SAND_DARK, mark="random", markrgb=SAND_MARK, random=0.4,
                width=512, height=512,
            ),
            sky,
        ),
        materials=(
            sc.MaterialCfg(
                name="seabed_sand", texture="seabed_sand", texuniform=True,
                texrepeat=(2.0, 2.0), reflectance=0.0, geom_names_expr=GROUND,
            ),
        ),
        lights=lights,
        terrain_type="plane",
        events={"granular_drag": _drag_event()},
        decorate=_decorate,
    )
