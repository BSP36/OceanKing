# OceanKing physics specification

The underwater physics every OceanKing experiment runs on. **This document is the
specification**: code that disagrees with it is a bug in one or the other, and a
change to either is made in both, in the same commit.

Implementation: [`oceanking/hydro.py`](../oceanking/hydro.py) (configuration, current),
[`oceanking/buoyancy.py`](../oceanking/buoyancy.py) (buoyancy, ballast kit, added mass),
[`oceanking/crown.py`](../oceanking/crown.py) (crown geometry),
[`oceanking/drag.py`](../oceanking/drag.py) (drag calibration). Measurements:
[`oceanking/tools/link_volumes.py`](../oceanking/tools/link_volumes.py),
[`drag_calibration.py`](../oceanking/tools/drag_calibration.py),
[`physics_report.py`](../oceanking/tools/physics_report.py) (prints the numbers below).
Verification: [`tests/`](../tests).

Status: implemented and verified as described below, 2026-10-25 (section 12 lists
what the 2026-10-25 review changed).

## 1. Conventions

- SI units: m, kg, s, N.
- World frame: right-handed, z up, gravity `g = (0, 0, -9.81) m/s^2`.
- Body frames are MuJoCo's. Jumper's base frame: x forward, y left, z up.
- Quaternions are MuJoCo's order, (w, x, y, z).
- Water: fresh, `rho = 1000 kg/m^3`, dynamic viscosity `mu = 1.0e-3 Pa s`. Declared
  scenario values, not calibrated ones.
- `beta` (buoyancy ratio) = buoyant force / weight = `rho V / M`, with `M` the
  physical mass. `beta < 1` sinks, `beta = 1` is neutral, `beta > 1` floats.

## 2. Scope

The robot is **fully submerged** and resting on or walking over a rigid seabed.
Modelled: gravity, buoyancy with its centre, drag, added mass, a uniform constant
current. Not modelled: a water surface or partial submersion, lift, waves,
turbulence, the seabed boundary layer, sediment, pressure effects on the hull,
fluid–structure coupling. Fully submerged and incompressible, the forces do not
depend on depth; **depth is a rendering parameter only** (`--depth` of the `seabed`
scene).

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
volume of the union. The per-link table
[`oceanking/data/jumper_link_volumes.json`](../oceanking/data/jumper_link_volumes.json)
holds each link's credited volume and its **centre of buoyancy** (the centroid of
the credited part, in the link's frame), is stamped with the model's SHA-256, and
is refused for any other model.

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
volume is **3–5 L, about 4 L to one significant figure**. Convex hulls overstate
concave parts, so 3.42 L is if anything high. The model's overall envelope at
HOME, 368 x 415 x 176 mm, agrees with the official 400 x 400 x 200 mm within 10%,
so the model is to scale.

### 3.3 Consequence

Sealed, `beta = 1000 * 3.42e-3 / 2.543 = 1.35` (with the product's 1.8 kg it would
be 1.9–2.7). **Jumper as built floats.** Walking on the seabed requires ballast:
section 5.

## 4. Gravity and buoyancy

Gravity is MuJoCo's, on the physical masses; it is never scaled, and no mass is
reduced to imitate buoyancy.

For every link `i` with credited volume `V_i`, an upward force `rho V_i g` acts
at the link's **centre of buoyancy**. MuJoCo's gravity compensation acts only at a
body's centre of mass, so each link carries a welded *buoyancy point*: a child
body at the centroid with mass `epsilon = 1e-5 kg` and

```text
gravcomp_point = 1 + rho V_i / epsilon      (lifts rho V_i g net of its own weight)
gravcomp_link  = 0
```

The 41 points add 0.41 g in total; their drag is negligible (section 6). Because
volume and mass are distributed differently (the upper shell is 0.10 kg and
0.56 L; the base 0.89 kg and 1.24 L), the forces produce the righting and
capsizing moments a real hull feels.

A diagnostic `uniform` model (`--buoyancy-model uniform`) gives every body the same
`gravcomp = beta`: the whole buoyant force at the centre of mass, no moment, no
ballast kit. It exists to isolate effects, not to describe a robot.

## 5. The OceanKing ballast kit

Mounted on the base link, welded (no joints), so the policy's observation and
action are unchanged.

| Part | Where | Specification | Buoyancy |
|---|---|---|---|
| **Keel** | inside the torso, 5 mm above its floor, placed to trim (below) | lead (11340 kg/m^3), 120 x 100 mm footprint, thickness from its mass | none of its own: inside the sealed hull |
| **Crown** | on the torso roof, centred, seated 3 mm into its crest | gold band (r = 65 mm, h = 45 mm) round a velvet dome (half-ellipsoid, r = 61 mm, h = 45 mm), five gold points, an orb; the band and dome are one sealed PETG tank, 3 mm wall: outer volume 0.977 L, tank 0.803 L, dry mass 0.221 kg | `rho * 0.977 L`, at the tank's centroid |

The crown's ballast tank holds air fraction `a` (`--crown-air`, 0 flooded to 1
blown); its water, `rho * 0.803 L * (1 - a)`, is crown mass. The **keel mass is
solved** so that the whole robot has the requested `beta` at that `a`:

```text
M_keel = rho (V_robot + V_crown) / beta - (m_robot + m_crown_dry + rho V_tank (1 - a))
```

and the **keel position is solved** so that, at HOME, the whole robot's centre of
mass lies directly below its centre of buoyancy (no static pitch or roll moment),
as a real vehicle is trimmed. Values at the default `a = 0.5`, with the `beta`
range the tank can then reach (`a` from 0 to 1):

| beta | keel | plate | keel centre (x, y) | total mass | tank trim range |
|---|---|---|---|---|---|
| 0.70 | 3.122 kg | 22.9 mm | (+1.0, +0.3) mm | 6.287 kg | 0.658 – 0.748 |
| **0.80** (default) | **2.336 kg** | **17.2 mm** | **(+0.1, +0.4) mm** | **5.501 kg** | **0.746 – 0.863** |
| 0.90 | 1.724 kg | 12.7 mm | (−1.2, +0.4) mm | 4.890 kg | 0.832 – 0.981 |
| 0.95 | 1.467 kg | 10.8 mm | (−2.0, +0.4) mm | 4.633 kg | 0.874 – 1.040 |

Refused: a keel thicker than half the torso's height (79 mm), i.e. `beta` below
about 0.55; a negative keel (`beta` above about 1.6 with the tank blown); a
trimming position outside the torso.

**Stability** (beta 0.8, HOME): centre of mass 118.8 mm, centre of buoyancy
130.0 mm above the ground; `BG = 11.3 mm`, a righting moment of 0.49 N m per
radian of roll or pitch. The crown is the buoyancy tank rather than a weight on
purpose: a heavy crown would put the centre of mass high.

The tank fill is constant within a run. Changing it at run time (diving,
surfacing, trimming) is the next step and will keep the keel fixed.

**Collision.** Every part of the crown that stands proud — band, dome, the five
points and their tips, the orb — collides exactly as the torso does: its
collision bits are copied from the torso's colliding geom at build time, so it
meets the ground and props and not the robot's own limbs. The jewels are set
flush into the band and are visual only. The keel has no collision: it lies
inside the torso's hull, which meets anything first.

**Seat.** The roof is curved; the crown sits 3 mm below its crest so that no gap
shows at the rim. The overlap with the torso's hull, at most 40 cm^3 (1.2% of the
robot's volume), is counted as displacement twice.

## 6. Drag and current

### 6.1 Model

MuJoCo's **ellipsoid fluid model** on every colliding geom of the robot, plus the
crown. Each geom is replaced by the ellipsoid with its inertia (uniform density),
and the force on it is computed from the velocity relative to the water: blunt
drag (coefficient 0.5), slender drag (0.25), angular drag (1.5), and viscous
(Stokes) drag; **Kutta lift and Magnus force are off**. The keel plate carries the
fluid flag with all coefficients zero (inside the hull, no drag of its own); the
buoyancy points are left on MuJoCo's inertia-box fallback, which at their size
(0.8 mm) gives all 41 together under 0.1% of the robot's drag.

**Why the ellipsoid model.** The inertia-box model builds each body's box from its
inertia tensor, i.e. from how mass is distributed; lead inside the torso would
change the torso's drag with no change of shape. The ellipsoid model reads
geometry only.

### 6.2 Calibration

Summed geom by geom, with overlapping hulls both exposed and nothing shielded,
MuJoCo's model overstates the robot's drag about threefold. It is calibrated
against the robot's **silhouette** ([`oceanking/drag.py`](../oceanking/drag.py),
[`data/jumper_drag.json`](../oceanking/data/jumper_drag.json)): the standing,
kitted robot at HOME in a 0.3 m/s flow, silhouette by rasterising the union of
its colliding geoms' outlines at 1 mm.

| Flow | Silhouette | Raw per-geom Cd | Calibrated Cd |
|---|---|---|---|
| head-on (x) | 551 cm^2 | 3.10 | **1.00** (target) |
| side-on (y) | 422 cm^2 | 4.01 | 1.29 |
| heave (z) | 692 cm^2 | 2.82 | 0.91 |

One factor, **0.3226**, multiplies the density (and viscosity) the drag model
sees, setting head-on Cd to 1.0; the directional ratios stay MuJoCo's. Rough
bluff bodies at Reynolds number ~1e5 (0.3 m/s over 0.3 m) — open-frame ROVs,
cubes, short cylinders in cross-flow — have Cd of about 0.8–1.3; all three
calibrated values fall in it. The file is stamped with the model's hash and the
crown's volume and refused when either changes.

Resulting drag, standing robot: **2.5 N at 0.3 m/s**, head-on or side-on.

### 6.3 Current

`option.wind`: uniform, constant, horizontal, world frame (`--current VX VY`). Drag
acts on `v_body - v_current`, so a body at rest in a current is pushed downstream
and a body moving with it feels nothing. The value is the water speed **at the
robot's height**; a real near-bed current is slower than the free stream.

`--drag-scale k` multiplies the calibrated drag, for sensitivity studies (Cd 0.8
and 1.3 are `k` = 0.8 and 1.3). Buoyancy does not depend on it.

## 7. Added mass

A body accelerating in water accelerates water with it. MuJoCo's fluid models
include only the velocity-dependent added-mass terms, so the acceleration term is
added here: each link and the crown gain `dm = C_a rho V` of mass, their inertia
is scaled by `(m + dm) / m`, and the extra weight is compensated
(`gravcomp = (gravcomp m + dm) / (m + dm)`). Net weight, buoyancy, contact loads and
the static balance are unchanged; only inertia grows. `C_a = 0.5` (`--added-mass`),
the sphere's value; streamlined bodies have less, plates and cylinders across the
flow up to about 1. Applied with drag (modes `drag` and `full`).

For beta 0.8: 2.20 kg of added mass, 40% of the 5.50 kg robot; the free sink
acceleration drops from 1.96 to 1.40 m/s^2, and the terminal sink speed against
heave drag is 0.59 m/s.

**Approximation:** isotropic, at each link's centre of mass. Real added mass is a
direction-dependent tensor, largest for motion across a body's flat faces.

## 8. The seabed (scenes)

The ground belongs to the **scene** (`--scene`), the water to the task, so each
ground can be run wet and dry. Friction is set on the ground with a contact
priority above the feet's (upstream's `set_ground_contact`), so the ground's
value is the one the solver uses and the feet's dry 1.2 is discarded; the feet's
contact stiffness is kept unless the ground is soft.

| Scene | Ground | Sliding friction | Contact | Basis |
|---|---|---|---|---|
| `seabed` | flat, hard | feet's 1.2 (unchanged) | feet's | reference ground, not a claim about any real seabed |
| `seabed_rock` | heightfield, 20 mm peak-to-peak (3.8 mm rms), 0.2 m stones, fixed seed | **0.5** | feet's | rubber on wet rock: ~0.5–0.7 on rough clean faces, ~0.2–0.4 polished or with biofilm; wet rubber friction is carried by hysteresis against texture, which a water or biological film over smooth rock removes |
| `seabed_sand` | flat, soft (feet sink ~1 cm on land) | **0.6** | upstream `soft`: compliant, plus granular drag on a sunk foot | a foot sliding on saturated sand shears the sand: tan of the internal friction angle, 30–35 deg for clean sand, is 0.58–0.70 |

Neither friction value is a measurement of this robot's feet; both are
literature-range choices and are the first parameters to vary in a sensitivity
study. The sand's granular drag is upstream's, which records that its constant has
no measured basis. The rock relief is 80% of the gait's 25 mm foot lift; the
official policy never trained on uneven ground and has no terrain sensing.

With these values the traction limit of section 11 becomes, at beta 0.8, a
relative water speed of about **0.44 m/s on rock** and **0.48 m/s on sand**
(`v = sqrt(2 mu (1 - beta) M g / (rho Cd A))`, head-on).

## 9. Configuration

`ocean.posture` (`jumper.posture` with this physics; same observation, action and
rewards). Each option falls back to an environment variable, then to the default.

| Option | Env variable | Default | Meaning |
|---|---|---|---|
| `--hydro {off,buoyancy,drag,full}` | `OCEAN_HYDRO` | `full` | `off`: the unmodified land robot; `buoyancy`: buoyancy and kit; `drag`: drag, current, added mass; `full`: all |
| `--buoyancy BETA` | `OCEAN_BUOYANCY` | 0.8 | whole-robot buoyancy ratio |
| `--buoyancy-model {archimedes,uniform}` | `OCEAN_BUOYANCY_MODEL` | `archimedes` | section 4 |
| `--crown-air A` | `OCEAN_CROWN_AIR` | 0.5 | tank air fraction the keel is sized at |
| `--current VX VY` | `OCEAN_CURRENT` | 0 0 | m/s, world frame |
| `--drag-scale K` | `OCEAN_DRAG_SCALE` | 1 | multiplier on the calibrated drag |
| `--added-mass CA` | `OCEAN_ADDED_MASS` | 0.5 | added-mass coefficient |
| `--nominal` | `OCEAN_NOMINAL` | off | replays: start at the origin facing +x, no startup randomisation (foot friction 1.2, no COM offset, no encoder bias) |
| `--trajectory-out PATH` | `OCEAN_TRAJECTORY_OUT` | — | replays: record root pose and command every control step |

Every effect is a model edit made before compilation: MuJoCo applies it in every
physics substep (1 kHz) on both backends, with nothing to clear on reset.
`--hydro off` leaves the robot exactly as upstream builds it. Training, replay and
evaluation of one experiment must be given the same values; replays print the
resolved condition and `--trajectory-out` stores it.

## 10. Verification

`tests/` (`.venv/bin/python -m pytest tests/`), each test naming the silent failure
it guards against. On a box (MuJoCo Warp on the GPU and native MuJoCo):

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
| Net weight of the compiled robot | `(1 - beta) M g` (rel. 1e-6) for several `beta`, `a` |
| Each link's buoyancy | `rho V_i g`, at the stored centroid, link itself uncompensated |
| Inertial frames of the kit and buoyancy points | as intended (see section 12, item 3) |
| Trim | horizontal offset of centre of buoyancy from centre of mass < 1 µm at beta 0.7, 0.8, 0.95 |
| Standing at HOME under the policy's PD gains | ground load `(1 - beta) M g` (within 0.2% of weight) at beta 0.8 and 0.95 |
| Whole robot, free water, no drag | centre of mass accelerates at `-(1 - beta) g` on both simulators |
| Added mass | sink acceleration `(1 - beta) M g / (M + C_a rho V)` on both simulators; inertia grows, net weight does not |
| Rolled 30 deg | restoring moment with the kit; uniform model has none beyond `(1 - beta)` of land |
| Dropped upside down (buoyancy zeroed) | rests on the crown; control without crown collision sinks into the floor |
| Drag calibration | stored factor reproduced by a fresh measurement; all three Cd in 0.8–1.3; stale file refused; factor reaches `option.density` |
| Buoyancy points and keel | keel flagged with zero drag; points' fallback drag under 0.1% of the robot's |
| Through the real task and scene path | `beta`, kit, drag flags, density and current reach the compiled model under any `--scene` |
| Seabed grounds | rock and sand outrank the feet with friction 0.5 / 0.6 (rock: feet's stiffness; sand: soft); the reference seabed does not |

## 11. Measurements

Workstation: RTX PRO 6000 Blackwell, Ryzen 9 9950X3D; MuJoCo 3.11.0, MuJoCo Warp
3.11.0.

**Physical quantities** (`physics_report`, beta 0.8, crown air 0.5): weight 53.97 N,
buoyancy 43.17 N, net 10.79 N; BG 11.3 mm; head-on drag 2.48 N at 0.3 m/s;
added mass 2.20 kg; terminal sink 0.59 m/s; traction `mu (1 - beta) M g` = 12.95 N
with the nominal foot friction 1.2, against which the head-on drag balances at a
relative water speed of **0.69 m/s** (beta 0.9: 0.46 m/s; beta 0.95: 0.31 m/s).
Section 8 gives the same limit on wet rock and sand.

**Training cost**, `ocean.posture`, 4096 environments, per PPO iteration: land
1.31 s; water 2.00 s (2026-10-25). History: inertia-box drag 1.36 s; ellipsoid
drag 1.74 s; crown with 13 colliding geoms 1.83 s; buoyancy points 2.00 s (with a
zero-drag marker geom on each point it was 2.38 s, which is why they have none).

**Righting** (2026-10-25, native, beta 0.8, joints PD-held at HOME, no policy):
dropped upside down from 0.35 m, the robot lands on its crown and the kit's
moment turns it from inverted (body z-axis . world z = −1.00) to 0.93 (21 deg from
upright) within 2.5 s.

**Official `jumper.posture` policy, land vs water** (2026-10-25, `--nominal`, one
run each, native backend, command script: forward 0.5 m/s, then sideways 0.3 m/s,
then turn). Mean velocity over the steady part of each phase:

| Condition | forward phase (cmd 0.5) | sideways phase (cmd 0.3) |
|---|---|---|
| Land | 0.42 m/s | 0.22 m/s |
| Water, beta 0.8 | 0.33 m/s | 0.22 m/s |
| Water, beta 0.95 | 0.17 m/s | 0.16 m/s |
| Water, beta 0.8, head current 0.3 m/s | 0.16 m/s | 0.21 m/s, drifting −0.05 m/s in x |

Repeat runs of the same nominal condition agree to about 2 cm in final position.
Not yet separated: how much of the slow-down is drag, reduced traction, or the
kit's mass and added mass.

**Official policy on the seabed grounds** (2026-10-25, `--nominal`, one run each,
same command script; "hold" is the last 4 s, commanded to stand still; m/s):

| Condition | forward (cmd 0.5) | sideways (cmd 0.3) | hold | final position |
|---|---|---|---|---|
| Land, flat | 0.42 | 0.22 | 0.00 | (1.60, 0.72) m |
| Land, rock ground | 0.45 | 0.23 | 0.00 | (1.81, 0.71) m |
| Land, sand ground | 0.36 | 0.19 | 0.00 | (1.32, 0.65) m |
| Water beta 0.8, flat | 0.33 | 0.22 | 0.00 | (1.14, 0.69) m |
| Water beta 0.8, rock | 0.28 | 0.21 | 0.00 | (1.13, 0.54) m |
| Water beta 0.8, sand | 0.25 | 0.19 | −0.01 | (0.84, 0.56) m |
| Water, rock, head current 0.3 | **0.07** | 0.15 | **−0.07** | **(−0.30, 0.34) m** |
| Water, sand, head current 0.3 | **0.08** | 0.18 | **−0.07** | **(−0.46, 0.50) m** |
| Water, rock, cross current 0.3 | 0.25 (drift −0.13) | **0.06** | −0.07 | (0.98, −0.68) m |
| Water beta 0.9, rock, cross current 0.2 | 0.17 (drift −0.13) | **0.01** | −0.07 | (0.70, −0.87) m |

It never fell (body z-axis never below 0.99, no resets). It **fails by being
carried away**: against a 0.3 m/s head current on rock or sand it ends behind its
start, and it cannot stand still in a current, losing 0.07 m/s downstream while
commanded to hold. Standing on all six feet would hold: on rock at beta 0.8 the
head-on drag at 0.3 m/s is 2.5 N against 5.4 N of traction. The policy keeps
stepping in place, and each lifted foot gives up its share.

## 12. Review of 2026-10-25

Gravity, mass and buoyancy were re-derived from scratch and drag compared with
an independent estimate. Changed:

1. **Drag was about three times too high** (Cd 3.1 head-on): calibrated to Cd 1.0
   (section 6.2). The land-vs-water slow-down measured before was exaggerated by it.
2. **Centre of buoyancy at each link's centre of mass** overstated stability: BG
   was 14.2 mm where the hull centroids give 9.7 mm (keel then mis-placed, item
   3). Buoyancy now acts at the centroids (section 4).
3. **Explicit inertials left `ipos` unset.** MjSpec's default is NaN and the
   compiler then sets the centre of mass to the body's position again, doubling
   the offset: the keel's centre of mass sat twice as far from the torso origin
   as intended, which lowered stability and shifted trim. Every added body now
   sets `ipos`, and a test pins it.
4. **No trim**: the centre of buoyancy lay 10 mm ahead of the centre of mass, a
   standing 0.43 N m nose-up moment. The keel is now placed to cancel it.
5. **Added mass was missing**: about 40% of the robot's mass (section 7).

## 13. Known limitations and open items

1. The simulated robot is 2.543 kg; the product is 1.8 kg (section 3.1).
2. Volumes are convex hulls of collision meshes, every link sealed (section 3.2).
3. Drag: one calibration factor; per-direction ratios are MuJoCo's per-geom sums,
   with no shielding between parts; no lift (section 6).
4. Added mass isotropic, at link centres of mass (section 7).
5. The kit is welded; the crown does not collide with the robot's own limbs, and
   its seat overlaps the hull by up to 40 cm^3 (section 5).
6. Fully submerged only; no water surface; current uniform down to the seabed.
7. Friction on rock (0.5) and sand (0.6) are literature-range choices, not
   measurements (section 8); the reference `seabed` keeps the dry 1.2.
