"""seabed -- a flat sandy seabed seen through water.

**A look, not physics.** The ground is upstream's flat plane with its contact
parameters untouched; only textures, lights and the sky change. The water's
forces come from the task (`ocean.posture --hydro ...`), never from a scene, so
the same scene can show the land baseline and the underwater runs side by side.

Rock, sand and seaweed variants are meant to live beside this module, sharing
`underwater_light`, and to register in `oceanking.plugin.install_scenes`.

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

SAND = (0.76, 0.70, 0.55)
SAND_MARK = (0.60, 0.55, 0.42)


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
                name="seabed_sand", type="2d", builtin="flat",
                rgb1=SAND, rgb2=SAND, mark="random", markrgb=SAND_MARK, random=0.35,
                width=512, height=512,
            ),
            sky,
        ),
        materials=(
            sc.MaterialCfg(
                name="seabed_sand", texture="seabed_sand", texuniform=True,
                texrepeat=(3.0, 3.0), reflectance=0.0, geom_names_expr=GROUND,
            ),
        ),
        lights=lights,
    )
