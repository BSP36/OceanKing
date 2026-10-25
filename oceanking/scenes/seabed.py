"""seabed -- a flat, hard seabed seen through water: the reference ground.

**A look, not physics.** The ground is upstream's flat plane with its contact
parameters untouched; only textures, lights and the sky change. The water's
forces come from the task (`ocean.posture --hydro ...`), never from a scene, so
the same scene can show the land baseline and the underwater runs side by side.

Rock, sand and seaweed variants are meant to live beside this module, sharing
`underwater_light`, and to register in `oceanking.plugin.SCENES`.

## Landmarks

So that distance travelled can be read off a picture: the sand carries a faint
0.5 m checkerboard, and survey stakes stand every 0.5 m along the +x and +y
axes from the world origin, every metre's stake taller and red. They are visual
only (no collision) and stand around the origin, i.e. around environment 0 --
what a single-environment replay shows.

## Depth

`--depth` only changes the picture. Sunlight entering water loses red first, then
green; the light colour here is the surface sunlight times
``exp(-ATTENUATION * depth)`` per channel, renormalised so the brightest channel
keeps `LIGHT_LEVEL` (an exposure correction, as a camera would make). The
attenuation coefficients are rough clear-coastal-water values, chosen for look.
"""

from __future__ import annotations

import math

from scenes.registry import GROUND, Headlight, Scene

DEFAULT_DEPTH_M = 5.0

#: Per-channel diffuse attenuation, 1/m, red / green / blue.
ATTENUATION = (0.35, 0.07, 0.04)

#: Brightest channel of the directional light after renormalising.
LIGHT_LEVEL = 1.2

#: A hard, clean floor (grey-blue), so it is not mistaken for `seabed_sand`.
SAND = (0.62, 0.64, 0.66)
SAND_DARK = (0.52, 0.54, 0.57)
SAND_MARK = (0.45, 0.47, 0.50)

#: Checker cell edge, m. With texuniform, texrepeat is repeats per metre and a
#: builtin checker draws one cell per repeat (measured against the stakes).
GRID_CELL = 0.5
#: Survey stakes: spacing and how far out along +x and +y.
STAKE_SPACING = 0.5
STAKE_REACH_X = 3.0
STAKE_REACH_Y = 2.0


#: How much of the attenuated colour reaches the picture. 1.0 is the physical
#: colour cast, which at a few metres turns everything cyan and the red robot
#: black; a diver's camera white-balances most of it away.
TINT_STRENGTH = 0.45


def underwater_tint(depth: float) -> tuple[float, float, float]:
    """Relative colour of sunlight at `depth` metres, brightest channel 1."""
    raw = [math.exp(-k * max(depth, 0.0)) for k in ATTENUATION]
    top = max(raw)
    return tuple(1.0 - TINT_STRENGTH * (1.0 - r / top) for r in raw)


def underwater_light(depth: float):
    """Lights, headlight and sky colours for water `depth` metres deep."""
    from mjlab.utils import spec_config as sc

    tint = underwater_tint(depth)
    dim = math.exp(-0.04 * max(depth, 0.0))  # overall loss, kept gentle
    sun = tuple(LIGHT_LEVEL * dim * c for c in tint)
    ambient = tuple(0.35 * dim * c for c in tint)
    water = underwater_tint(depth * 4.0)  # the water column itself looks bluer
    sky_top = tuple(0.45 * dim * c * b for c, b in zip(water, (0.35, 0.75, 1.0)))
    sky_bottom = tuple(0.06 * c * b for c, b in zip(water, (0.35, 0.75, 1.0)))
    lights = (
        sc.LightCfg(
            name="sun", type="directional", pos=(0.0, 0.0, 6.0),
            dir=(0.25, 0.15, -1.0), castshadow=True,
            diffuse=sun, specular=(0.1, 0.1, 0.1), ambient=(0.0, 0.0, 0.0),
        ),
    )
    headlight = Headlight(ambient=ambient, diffuse=(0.0, 0.0, 0.0))
    sky = sc.TextureCfg(
        name="seabed_sky", type="skybox", builtin="gradient",
        rgb1=sky_top, rgb2=sky_bottom, width=512, height=3072,
    )
    return lights, headlight, sky


def cli_args(group):
    """`--depth`, see the module docstring. Falls back to `OCEAN_DEPTH`."""
    import os

    group.add_argument(
        "--depth", dest="depth", type=float,
        default=float(os.environ.get("OCEAN_DEPTH", "").strip() or DEFAULT_DEPTH_M),
        metavar="M",
        help=f"water depth in metres, for the light only (default: {DEFAULT_DEPTH_M:g})",
    )
    return ("depth",)


def scene(depth: float = DEFAULT_DEPTH_M) -> Scene:
    from mjlab.utils import spec_config as sc

    lights, headlight, sky = underwater_light(depth)
    return Scene(
        headlight=headlight,
        textures=(
            sc.TextureCfg(
                name="seabed_sand", type="2d", builtin="checker",
                rgb1=SAND, rgb2=SAND_DARK, mark="random", markrgb=SAND_MARK, random=0.15,
                width=512, height=512,
            ),
            sky,
        ),
        materials=(
            sc.MaterialCfg(
                name="seabed_sand", texture="seabed_sand", texuniform=True,
                texrepeat=(1.0 / GRID_CELL,) * 2, reflectance=0.0,
                geom_names_expr=GROUND,
            ),
        ),
        lights=lights,
        decorate=add_landmarks,
    )


def add_landmarks(spec) -> None:
    """Survey stakes along +x and +y, and a disc at the origin. Visual only."""
    import mujoco

    def stake(x: float, y: float, metre: bool) -> None:
        h = 0.30 if metre else 0.16
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.015, h / 2, 0], pos=[x, y, h / 2],
            rgba=[1.0, 0.25, 0.1, 1] if metre else [1.0, 1.0, 1.0, 1],
            contype=0, conaffinity=0, group=0,
        )

    def ticks(reach: float) -> list[float]:
        n = int(round(reach / STAKE_SPACING))
        return [STAKE_SPACING * k for k in range(1, n + 1)]

    for x in ticks(STAKE_REACH_X):
        stake(x, -0.4, abs(x - round(x)) < 1e-9)
    for y in ticks(STAKE_REACH_Y):
        stake(-0.4, y, abs(y - round(y)) < 1e-9)
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.25, 0.001, 0], pos=[0, 0, 0.001],
        rgba=[0.2, 0.6, 0.9, 0.6], contype=0, conaffinity=0, group=0,
    )
