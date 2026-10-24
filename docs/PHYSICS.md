# OceanKing physics specification

The underwater physics every OceanKing experiment runs on. **This document is the
specification**: code that disagrees with it is a bug in one or the other, and a
change to either is made in both, in the same commit.

Implementation: [`oceanking/hydro.py`](../oceanking/hydro.py) (configuration, drag,
current), [`oceanking/buoyancy.py`](../oceanking/buoyancy.py) (Archimedes buoyancy,
ballast kit), [`oceanking/tools/link_volumes.py`](../oceanking/tools/link_volumes.py)
(volume measurement). Verification: [`tests/`](../tests).

Status: implemented and verified as described below, 2026-10-24.

## 1. Conventions

- SI units: m, kg, s, N.
- World frame: right-handed, z up, gravity `g = (0, 0, -9.81) m/s^2`.
- Body frames are MuJoCo's. Jumper's base frame: x forward, y left, z up.
- Quaternions are MuJoCo's order, (w, x, y, z).
- Water: fresh, `rho = 1000 kg/m^3`, dynamic viscosity `mu = 1.0e-3 Pa s`. Declared
  scenario values, not calibrated ones.
- `beta` (buoyancy ratio) = buoyant force / weight = `rho V / M`. `beta < 1` sinks,
  `beta = 1` is neutral, `beta > 1` floats.

## 2. Scope

The robot is **fully submerged** and resting on or walking over a rigid seabed.
Modelled: buoyancy, drag, a uniform constant current. Not modelled: a water
surface or partial submersion, added mass, lift, waves, turbulence, sediment,
pressure effects on the hull, any fluid–structure coupling. Fully submerged and
incompressible, the forces do not depend on depth; **depth is a rendering
parameter only** (`--depth` of the `seabed` scene).

## 3. The robot: mass and volume

### 3.1 Mass

| Source | Mass |
|---|---|
| Official product page and upstream `docs/HARDWARE.md` | 1.8 kg (engineering prototype 2.8 kg) |
| Upstream simulation model `jumper/assets/jumper/jumper.xml` (sum of 41 links) | **2.543 kg** |

**Decision: the simulation uses the model's 2.543 kg.** The official
`jumper.posture` policy was trained on this model, and the land-to-water
comparison needs the robot the policy knows. The 0.74 kg (41%) difference from
the product is a stated limitation of the upstream model, not corrected here.

### 3.2 Volume

Each link is treated as a **sealed** housing: it displaces its outer volume, air
inside. A link's volume is the convex hull of its collision mesh (`group=1` geom
of `jumper.xml`). Neighbouring hulls overlap, so each link is credited only with
the part of its hull no earlier link (body order, torso first) covers, estimated
by Monte Carlo inside the hull at the HOME pose. The credited volumes sum to the
volume of the union; the per-link table is
[`oceanking/data/jumper_link_volumes.json`](../oceanking/data/jumper_link_volumes.json),
stamped with the model's SHA-256 and refused for any other model.

| Part | Volume |
|---|---|
| Torso (base + upper shell) | 1.80 L |
| Four walking legs | 0.95 L |
| Two arms with claws | 0.67 L |
| **Total (union of hulls)** | **3.42 L** (sum of hulls before overlap removal: 4.35 L) |

Seed-to-seed spread of the Monte Carlo total: 0.05%.

**Cross-check by primitives** (torso as a box, limbs as straight cylinders, sizes
from the model at HOME): torso box 205 x 181 x 79 mm = 2.93 L; legs 4 x (222 mm,
dia. 40 mm) = 1.11 L; arms 2 x (327 mm, dia. 40 mm) = 0.83 L; total **4.9 L**. The
box ignores the torso's rounding and overestimates; limbs agree within 20%. The
volume is **3–5 L, about 4 L to one significant figure**. The model's overall
envelope at HOME, 368 x 415 x 176 mm, agrees with the official 400 x 400 x 200 mm
within 10%, so the model is to scale.

### 3.3 Consequence

Sealed, `beta = 1000 * 3.42e-3 / 2.543 = 1.35` (with the product's 1.8 kg it would
be 1.9–2.7). **Jumper as built floats.** Walking on the seabed requires ballast:
section 5.

## 4. Buoyancy

For every link `i` with credited volume `V_i` and mass `m_i`, a force
`F_B,i = -rho V_i g` (upward) acts at the link's centre of mass. It is applied as
MuJoCo gravity compensation:

```text
gravcomp_i = rho * V_i / m_i
```

Gravity, masses and inertias are untouched. Because volume and mass are not
distributed alike (the upper shell is 0.10 kg and 0.56 L; the base 0.89 kg and
1.24 L), the per-link forces produce the righting and capsizing moments a real
hull feels.

**Approximation:** each link's centre of buoyancy is placed at its centre of mass
rather than at its hull's centroid. Moments *between* links are represented;
the offset *within* one link is not.

A diagnostic `uniform` model (`--buoyancy-model uniform`) gives every body the same
`gravcomp = beta`: the whole buoyant force at the centre of mass, no moment, no
ballast kit. It exists to isolate effects, not to describe a robot.

## 5. The OceanKing ballast kit

Mounted on the base link, welded (no joints), so the policy's observation and
action are unchanged.

| Part | Where | Specification | Buoyancy |
|---|---|---|---|
| **Keel** | inside the torso, 5 mm above its floor, centred | lead (11340 kg/m^3), 120 x 100 mm footprint, thickness from its mass | none of its own: inside the sealed hull |
| **Crown** | on the torso roof, centred | sealed PETG cylinder, r = 65 mm, h = 75 mm, 3 mm wall: outer volume 0.995 L, tank 0.833 L, shell 0.206 kg | `rho * 0.995 L`, at the crown's centre of mass |

The crown's ballast tank holds air fraction `a` (`--crown-air`, 0 flooded to 1
blown); its water, `rho * 0.833 L * (1 - a)`, is crown mass. The **keel mass is
solved** so that the whole robot has the requested `beta` at that `a`:

```text
M_keel = rho (V_robot + V_crown) / beta - (m_robot + m_crown_shell + rho V_tank (1 - a))
```

Values at the default `a = 0.5`, and the `beta` range the tank can then reach
with that keel (`a` from 0 to 1):

| beta | keel | plate | total mass | tank trim range |
|---|---|---|---|---|
| 0.70 | 3.148 kg | 23.1 mm | 6.313 kg | 0.657 – 0.749 |
| **0.80** (default) | **2.359 kg** | **17.3 mm** | **5.524 kg** | **0.744 – 0.865** |
| 0.90 | 1.745 kg | 12.8 mm | 4.910 kg | 0.830 – 0.983 |
| 0.95 | 1.486 kg | 10.9 mm | 4.652 kg | 0.872 – 1.043 |

Refused: a keel thicker than half the torso's height (79 mm), i.e. `beta` below
about 0.55, and a negative keel (`beta` above about 1.6 with the tank blown).

**Stability.** Keel low and crown high put the centre of buoyancy above the centre
of mass, so a rolled robot is turned back upright (section 8). The crown is the
buoyancy tank rather than a weight on purpose: a heavy crown would put the centre
of mass high and make the robot capsize.

The tank fill is constant within a run. Changing it at run time (diving,
surfacing, trimming) is the next step and will keep the keel fixed.

**Not modelled:** the crown and keel do not collide with anything (visual geoms).

## 6. Drag and current

MuJoCo's **ellipsoid fluid model** on every colliding geom of the robot, plus the
crown. Each geom is replaced by the ellipsoid with its inertia (uniform density),
and the force on it is computed from the velocity relative to the water:

- blunt drag (coefficient 0.5), slender drag (0.25), angular drag (1.5), each
  proportional to `rho` — MuJoCo's defaults;
- viscous (Stokes) drag proportional to `mu`;
- **Kutta lift and Magnus force: off** (coefficients 0).

The keel carries the fluid flag with all coefficients zero: inside the hull, it
has no drag of its own.

The **current** is MuJoCo's `option.wind`: uniform, constant, horizontal, world
frame (`--current VX VY`). Drag acts on `v_body - v_current`, so a body at rest in
a current is pushed downstream and a body moving with it feels nothing.

`--drag-scale k` multiplies the density the drag model sees (`option.density =
k rho`); buoyancy does not use `option.density` and is unaffected.

**Why the ellipsoid model and not the inertia-box model.** The inertia-box model
builds each body's box from its inertia tensor, i.e. from how mass is distributed.
Lead inside the torso reshapes that tensor and so would change the torso's drag
with no change of shape. The ellipsoid model reads geometry only.

**Approximation:** overlapping hulls (base and upper shell) are both exposed to the
flow, so drag on the torso is overestimated; no occlusion or interaction between
parts is modelled.

## 7. Configuration

`ocean.posture` (`jumper.posture` with this physics; same observation, action and
rewards). Each option falls back to an environment variable, then to the default.

| Option | Env variable | Default | Meaning |
|---|---|---|---|
| `--hydro {off,buoyancy,drag,full}` | `OCEAN_HYDRO` | `full` | which effects are on; `off` is the unmodified land robot |
| `--buoyancy BETA` | `OCEAN_BUOYANCY` | 0.8 | whole-robot buoyancy ratio |
| `--buoyancy-model {archimedes,uniform}` | `OCEAN_BUOYANCY_MODEL` | `archimedes` | section 4 |
| `--crown-air A` | `OCEAN_CROWN_AIR` | 0.5 | tank air fraction the keel is sized at |
| `--current VX VY` | `OCEAN_CURRENT` | 0 0 | m/s, world frame |
| `--drag-scale K` | `OCEAN_DRAG_SCALE` | 1 | multiplier on drag |
| `--nominal` | `OCEAN_NOMINAL` | off | replays: start at the origin facing +x, no startup randomisation (foot friction 1.2, no COM offset, no encoder bias) |
| `--trajectory-out PATH` | `OCEAN_TRAJECTORY_OUT` | — | replays: record root pose and command every control step |

Every effect is a model edit made before compilation: MuJoCo applies it in every
physics substep (1 kHz) on both backends, with nothing to clear on reset.
`--hydro off` and `--hydro drag` leave the robot exactly as upstream builds it (no
kit). Training, replay and evaluation of one experiment must be given the same
values; replays print the resolved condition and `--trajectory-out` stores it.

## 8. Verification

`tests/` (run with `.venv/bin/python -m pytest tests/`), each test naming the
silent failure it guards against. On a box (both simulators, MuJoCo Warp on the
GPU and native MuJoCo):

| Check | Result |
|---|---|
| Free body, no drag | vertical acceleration `-g (1 - beta)` for beta 0–0.9 (rel. 1e-4) |
| Neutral buoyancy | no drift over 0.5 s |
| Fluid model alone | adds no buoyancy (no double counting with gravcomp) |
| Body at rest in a current | pushed downstream |
| Body moving with the current | no drag |
| Drag in still water | only removes energy; control without drag keeps its speed |
| Drag vs. mass distribution | ellipsoid model unchanged by internal ballast; control (inertia box) changes |
| Timestep 2 ms vs 1 ms | same velocity within 0.2% |
| `off` | bit-identical to the unmodified model |

On the real Jumper model:

| Check | Result |
|---|---|
| Volume table vs model | SHA-256 match required; masses must agree per link |
| Compiled whole-robot `beta` | equals the requested value (rel. 1e-9) for several `beta`, `a` |
| Each link's buoyant force | `rho V_i g` exactly |
| Standing at HOME under the policy's PD gains | ground load `(1 - beta) M g` (within 0.2% of weight) at beta 0.8 and 0.95 |
| Whole robot, free water, no drag | centre of mass accelerates at `-(1 - beta) g` on both simulators |
| Rolled 30 deg | restoring moment with the kit; uniform model has none beyond `(1 - beta)` of land |
| Through the real task and scene path | `beta`, kit, drag flags, density and current reach the compiled model under any `--scene` |

## 9. Measurements

Workstation: RTX PRO 6000 Blackwell, Ryzen 9 9950X3D; MuJoCo 3.11.0, MuJoCo Warp
3.11.0.

**Training cost**, `ocean.posture`, 4096 environments, per PPO iteration
(2026-10-24): land 1.31 s; water (`full`, ellipsoid drag) 1.74 s (+33%). The
earlier inertia-box drag cost 1.36 s; the ellipsoid model evaluates every
colliding geom.

**Official `jumper.posture` policy, land vs water** (2026-10-24, `--nominal`, one
run each, native backend, command script: forward 0.5 m/s, then sideways 0.3 m/s,
then turn). Mean velocity over the steady part of each phase:

| Condition | forward phase (cmd 0.5) | sideways phase (cmd 0.3) |
|---|---|---|
| Land | 0.42 m/s | 0.22 m/s |
| Water, beta 0.8 (kit) | 0.20 m/s | 0.18 m/s |
| Water, beta 0.95 (kit) | 0.09 m/s | 0.10 m/s |
| Water, beta 0.8, head current 0.3 m/s | **−0.07 m/s** (carried backwards) | 0.14 m/s, drifting −0.10 m/s in x |

Not yet separated: how much of the underwater slow-down is drag, how much is
reduced traction, and how much is the kit's added mass (5.5 kg against 2.5 kg).

## 10. Known limitations

1. The simulated robot is 2.543 kg; the product is 1.8 kg (section 3.1).
2. Volumes are hull volumes of collision meshes, treating every link as sealed.
3. Centre of buoyancy per link at the link's centre of mass (section 4).
4. Drag: ellipsoid approximation per geom, overlapping hulls double-exposed, no
   added mass, no lift (section 6).
5. The kit is welded and visual-only; the official policy was never trained with
   its mass.
6. Fully submerged only; no water surface.
