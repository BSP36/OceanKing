"""seabed_rock -- wet rock: an uneven rocky bed, slick with water and biofilm.

**This scene changes the physics** (ground shape and friction), like upstream's
`rough` and `ice`; the water itself still comes from the task.

## Friction: 0.5

Rubber on wet rock spans roughly 0.5-0.7 on rough, clean faces and falls to
0.2-0.4 on polished or biofilm-covered ones: wet rubber friction is carried by
hysteresis against surface texture, and a water or biological film over smooth
rock removes most of it. Silicone feet on sea-washed rock are taken at **0.5**,
between the two (docs/PHYSICS.md). The ground outranks the feet's contact
priority (`scenes.registry.set_ground_contact`), so this value replaces the feet's
dry 1.2; the contact stiffness stays the feet's own.

## Shape

A 24 m square field (inside a 12 m flat border, so no episode reaches an edge)
of a random heightfield, 20 mm peak-to-peak (3.8 mm rms) in cells of `CELL` --
stones a hand across, standing up to 80% of the 25 mm the gait lifts its feet. Built from
a fixed seed, so every run walks the same rocks. The official policy never saw
uneven ground and has no terrain sensing; that is part of what this scene tests.

The survey stakes and the origin disc of `seabed` stand on it as well.
"""

from __future__ import annotations

from scenes.registry import GROUND_PRIORITY, Scene

from .seabed import DEFAULT_DEPTH_M, add_landmarks, underwater_light
from .seabed import cli_args as _depth_args

#: Sliding, torsional, rolling. Torsional and rolling are the feet's own.
WET_ROCK_FRICTION = (0.5, 0.01, 0.01)
#: Upper end of the heightfield's noise, m. **Not the relief**: the generator's
#: quantisation and downsampling add 10 mm. Measured interior peak-to-peak relief
#: (std) for this seed: 10 mm -> 20.0 mm (3.8 mm); 15 -> 25.0 (5.1); 20 -> 30.0
#: (6.4). 20 mm is 80% of the gait's 25 mm foot lift (scenes/rough.py).
RELIEF = 0.010
#: Size of one stone, m (the heightfield's downsampled cell).
CELL = 0.2
#: One rock field, `TILE` square, inside a flat border `BORDER` wide. **Sized so
#: no episode can reach the edge**: no termination ends an episode when a robot
#: leaves the ground (upstream drops `out_of_terrain_bounds`), and found in review,
#: an 8 m tile with no border let robots walk off and fall forever. A training
#: episode is 20 s at up to 0.8 m/s in x and y at once (1.13 m/s), from a spawn
#: within 0.5 m: 23.1 m at most, against TILE/2 + BORDER = 24 m.
#: `tests/test_seabed_rock.py` checks this against the task's own numbers.
#: Rocks cover the first 12 m in every direction, where episodes spend their time;
#: a field this wide also keeps 4096 environments from sharing one patch of rock.
TILE = 24.0
BORDER = 12.0
SEED = 7
#: The generator names its ground `terrain_<n>`, which upstream's `GROUND` pattern
#: (`terrain$`, the plane) does not match -- found when `set_ground_contact`
#: raised on it. The friction and the rock material bind with this instead.
#:
#: The material is bound in `_decorate`, not through `Scene.materials`. Upstream
#: forbids a scene material under generated terrain because it would paint over
#: the generator's per-tile height colours (jumper/tests/test_scenes.py). Here the
#: generator colours nothing (`color_scheme="none"`): with one tile, its height
#: colour was a single flat green that showed neither relief nor distance.
ROCK_GROUND = (r"^terrain(_\d+)?$",)

ROCK = (0.46, 0.44, 0.40)
ROCK_DARK = (0.36, 0.35, 0.32)
ROCK_FILM = (0.25, 0.33, 0.22)  # green-brown biofilm flecks
#: Checker repeats across the tile: 0.5 m cells, like the other seabed scenes.
#: A heightfield maps one texture repeat over its whole extent.
ROCK_TEXREPEAT = TILE / 0.5
#: Contact and constraint budgets, per world. Upstream's rough scene's 131 / 1024
#: overflowed at construction on MuJoCo Warp ("nconmax must be >= 253", then
#: "njmax must be >= 1034"): the spawn state, root at the rock surface before the
#: first step, puts the whole robot in contact. Walking needs ~6 contacts.
NCONMAX = 320
NJMAX = 1400



def cli_args(group):
    return _depth_args(group)


def _terrain():
    from mjlab.terrains.heightfield_terrains import HfRandomUniformTerrainCfg
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

    return TerrainGeneratorCfg(
        size=(TILE, TILE), border_width=BORDER, num_rows=1, num_cols=1, seed=SEED,
        curriculum=False,
        color_scheme="none", difficulty_range=(1.0, 1.0),
        sub_terrains={"rock": HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.0, RELIEF), noise_step=0.005,
            downsampled_scale=CELL, horizontal_scale=0.05, vertical_scale=0.005,
            border_width=0.25,
        )},
    )


def _decorate(spec) -> None:
    """Wet-rock contact on every ground geom: upstream's `set_ground_contact`, with
    the generator's ground names. The feet's own stiffness is kept (priority
    replaces the whole parameter set), and matching nothing raises."""
    import re

    from tasks.jumper.common.constants import _CONTACT_SOLIMP, _CONTACT_SOLREF

    pats = [re.compile(p) for p in ROCK_GROUND]
    hits = 0
    for geom in spec.geoms:
        if any(p.search(geom.name or "") for p in pats):
            geom.friction = list(WET_ROCK_FRICTION)
            geom.solref = list(_CONTACT_SOLREF)
            geom.solimp = list(_CONTACT_SOLIMP)
            geom.priority = GROUND_PRIORITY
            hits += 1
    if not hits:
        raise RuntimeError(
            f"scene 'seabed_rock' matched no ground geom with {ROCK_GROUND!r}: the rock "
            "would have kept its own friction"
        )
    _rock_material(spec, pats)
    add_landmarks(spec)


def _rock_material(spec, pats) -> None:
    import mujoco

    spec.add_texture(
        name="seabed_rock", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=512, height=512,
        rgb1=list(ROCK), rgb2=list(ROCK_DARK), mark=mujoco.mjtMark.mjMARK_RANDOM,
        markrgb=list(ROCK_FILM), random=0.2,
    )
    mat = spec.add_material(name="seabed_rock", texrepeat=[ROCK_TEXREPEAT] * 2,
                            reflectance=0.05)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "seabed_rock"
    for geom in spec.geoms:
        if any(p.search(geom.name or "") for p in pats):
            geom.material = "seabed_rock"


def scene(depth: float = DEFAULT_DEPTH_M) -> Scene:
    lights, headlight, sky = underwater_light(depth)
    return Scene(
        headlight=headlight,
        textures=(sky,),
        materials=(),
        lights=lights,
        terrain_type="generator",
        terrain_generator=_terrain(),
        nconmax=NCONMAX,
        njmax=NJMAX,
        decorate=_decorate,
    )
