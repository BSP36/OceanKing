# OceanKing physics specification

The underwater physics every OceanKing experiment runs on. **This document is the
specification**: code that disagrees with it is a bug in one or the other, and a
change to either is made in both, in the same commit.

Implementation: [`oceanking/hydro.py`](../oceanking/hydro.py) (configuration, current),
[`oceanking/buoyancy.py`](../oceanking/buoyancy.py) (buoyancy, ballast kit, added mass),
[`oceanking/crown.py`](../oceanking/crown.py) (crown geometry),
[`oceanking/drag.py`](../oceanking/drag.py) (drag calibration),
[`oceanking/warp_patch.py`](../oceanking/warp_patch.py) (MuJoCo Warp fluid-torque fix),
[`oceanking/scenes/`](../oceanking/scenes) (seabed grounds). Measurements:
[`tools/link_volumes.py`](../oceanking/tools/link_volumes.py),
[`tools/drag_calibration.py`](../oceanking/tools/drag_calibration.py),
[`tools/physics_report.py`](../oceanking/tools/physics_report.py) (prints the numbers
below). Verification: [`tests/`](../tests).

Status: implemented and verified as described below, 2026-10-25. Section 12
records what two reviews of that day changed.

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
turbulence, the seabed boundary layer, sediment transport, pressure effects on the
hull, fluid–structure coupling. Fully submerged and incompressible, the forces do
not depend on depth; **depth is a rendering parameter only** (`--depth` of the
`seabed*` scenes).

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
the credited part, in the link's frame), is stamped with the XML's SHA-256, and is
refused for any other XML.

| Part | Volume |
|---|---|
| Torso (base + upper shell) | 1.80 L |
| Four walking legs | 0.95 L |
| Two arms with claws | 0.67 L |
| **Total (union of hulls)** | **3.42 L** (sum of hulls before overlap removal: 4.35 L) |

Seed-to-seed spread of the Monte Carlo total: 0.05%; an independent
re-measurement (6 M points, first-containing-hull assignment) gave 3.420 L.

**Uncertainty.** The collision meshes are slightly inside the visual ones: the
union of the *visual* hulls is 3.82 L (+11%). A cross-check by primitives (torso as
a box 205 x 181 x 79 mm, limbs as straight cylinders of 40 mm) gives 4.9 L, the box
overstating the rounded torso. The volume is **3.4–3.8 L, about 4 L to one
significant figure**. `beta` does not depend on it (the keel is solved for it);
the keel's mass and the stability margin do. The model's envelope at HOME,
368 x 415 x 176 mm, agrees with the official 400 x 400 x 200 mm within 10%.

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

The crown has one too, at its tank's centroid (section 5). The 42 points add
0.42 g in total and have no geom; MuJoCo's inertia-box drag fallback gives them
together under 0.1% of the robot's drag. Because volume and mass are distributed
differently (the upper shell is 0.10 kg and 0.56 L; the base 0.89 kg and 1.24 L),
the forces produce the righting and capsizing moments a real hull feels. An
independent review measured the restoring moment for roll 10–90 deg and found it
consistent with the BG below, and MuJoCo Warp and native MuJoCo agreeing to 8e-6 N.

A diagnostic `uniform` model (`--buoyancy-model uniform`) gives every body the same
`gravcomp = beta`: the whole buoyant force at the centre of mass, no moment, no
ballast kit. It exists to isolate effects, not to describe a robot.

## 5. The OceanKing ballast kit

Mounted on the base link, welded (no joints), so the policy's observation and
action are unchanged.

| Part | Where | Specification | Buoyancy |
|---|---|---|---|
| **Keel** | inside the torso, its underside 10 mm above the hull's lowest point, placed to trim (below) | lead (11340 kg/m^3), 110 x 90 mm footprint, thickness from its mass | none of its own: inside the sealed hull (every corner checked inside the base's hull at beta 0.6–0.95) |
| **Crown** | on the torso roof, centred, seated 3 mm into its crest | gold band (r = 65 mm, h = 45 mm) round a velvet dome (half-ellipsoid, r = 61 mm, h = 45 mm), five gold points, an orb; the band and dome are one sealed PETG tank, 3 mm wall: outer volume 0.977 L, tank 0.803 L, dry mass 0.221 kg | `rho * 0.977 L` at the tank's centroid (37.1 mm above the crown's base) |

The crown's ballast tank holds air fraction `a` (`--crown-air`, 0 flooded to 1
blown); its water, `rho * 0.803 L * (1 - a)`, is crown mass, **settled at the tank's
floor**: the crown's centre of mass is the dry shell at the tank's centroid
combined with the water's centroid (at `a = 0.5`: 25.8 mm above the crown's base,
against 37.1 mm for its buoyancy). The free surface of a part-filled tank (a
small loss of stability when tilted) is not modelled.

The **keel mass is solved** so that the whole robot has the requested `beta` at
that `a`:

```text
M_keel = rho (V_robot + V_crown) / beta - (m_robot + m_crown_dry + rho V_tank (1 - a))
```

and the **keel position is solved** so that, at HOME, the whole robot's centre of
mass lies directly below its centre of buoyancy (no static pitch or roll moment),
as a real vehicle is trimmed. Values at the default `a = 0.5`, with the `beta`
range the tank can then reach (`a` from 0 to 1):

| beta | keel | plate | keel centre (x, y) | total mass | tank trim range |
|---|---|---|---|---|---|
| 0.70 | 3.122 kg | 27.8 mm | (+1.0, +0.3) mm | 6.287 kg | 0.658 – 0.748 |
| **0.80** (default) | **2.336 kg** | **20.8 mm** | **(+0.1, +0.4) mm** | **5.501 kg** | **0.746 – 0.863** |
| 0.90 | 1.724 kg | 15.4 mm | (−1.2, +0.4) mm | 4.890 kg | 0.832 – 0.981 |
| 0.95 | 1.467 kg | 13.1 mm | (−2.0, +0.4) mm | 4.633 kg | 0.874 – 1.040 |

Refused: a keel thicker than half the torso's height (79 mm), i.e. `beta` below
about 0.55; a negative keel (`beta` above about 1.6 with the tank blown); a
trimming position outside the torso.

**Stability** (beta 0.8, HOME): centre of mass 120.4 mm, centre of buoyancy
130.0 mm above the ground; `BG = 9.6 mm`, a righting moment of 0.42 N m per
radian of roll or pitch (beta 0.9: 10.3 mm; 0.95: 10.5 mm). The crown is the
buoyancy tank rather than a weight on purpose: a heavy crown would put the centre
of mass high.

The tank fill is constant within a run. Changing it at run time (diving,
surfacing, trimming) is the next step and will keep the keel fixed.

**Collision.** Every part of the crown that stands proud — band, dome, the five
points and their tips, the orb — collides exactly as the torso does: its
collision bits are copied from the torso's colliding geom at build time, so it
meets the ground and props and not the robot's own limbs. The jewels are set
flush into the band and are visual only. The keel plate carries a collision bit
no geom's affinity matches: it touches nothing, but counts as colliding, so the
native backend's stripping of non-colliding geoms keeps it (and its fluid flag).

**Seat.** The roof is curved; the crown sits 3 mm below its crest so that no gap
shows at the rim. The overlap with the torso's hull, counted as displacement
twice, is 1.3 cm^3 (measured; the geometric bound is 40 cm^3).

## 6. Drag and current

### 6.1 Model

MuJoCo's **ellipsoid fluid model** on every colliding geom of the robot and on the
crown. Each geom is replaced by the ellipsoid with its inertia (uniform density),
and the force on it is computed from the velocity relative to the water: blunt
drag (coefficient 0.5), slender drag (0.25), angular drag (1.5), viscous (Stokes)
drag, and the model's velocity-dependent added-mass terms; **Kutta lift and Magnus
force are off**. Each geom's whole fluid force and torque is scaled by its
multiplier (`fluid_ellipsoid`), which is how the calibration below is applied; the
water's density stays 1000 kg/m^3. The keel plate's multiplier is 1e-9 (no fluid
force of its own, inside the hull), and other scene geoms are left alone.

**Why the ellipsoid model.** The inertia-box model builds each body's box from its
inertia tensor, i.e. from how mass is distributed; lead inside the torso would
change the torso's drag with no change of shape. The ellipsoid model reads
geometry only.

### 6.2 Calibration by part

Summed geom by geom, with no shielding, MuJoCo's model overstates drag unevenly:
the torso is two overlapping hulls and the crown a cluster of small parts, while a
leg alone is close to a cylinder. The robot is split into four groups, each with
its own multiplier ([`oceanking/drag.py`](../oceanking/drag.py),
[`data/jumper_drag.json`](../oceanking/data/jumper_drag.json)), chosen so that the
group alone, head-on at 0.3 m/s, has its target drag coefficient on its reference
area. Torso and crown are each one bluff body (reference: their silhouette); a
limb is a cylinder a few diameters from the next, which shields it little
(reference: the sum of each limb's own silhouette).

| Group | Geoms | Reference area | Raw Cd | Target Cd | Multiplier |
|---|---|---|---|---|---|
| torso | 5 | 125 cm^2 | 3.55 | 1.0 (rounded box; a cube is ~1.05) | 0.281 |
| crown | 13 | 113 cm^2 | 2.58 | 1.0 | 0.388 |
| legs | 16 (4 limbs) | 315 cm^2 | 1.48 | 1.1 (cylinder across the flow, Re ~1e4) | 0.743 |
| arms | 20 (2 limbs) | 194 cm^2 | 2.60 | 1.1 | 0.424 |

The groups, measured alone, sum to 97% of the whole (MuJoCo's per-body
aggregation is not exactly additive). Whole robot, calibrated, on its silhouette:

| Flow | Silhouette | Cd | Drag at 0.3 m/s |
|---|---|---|---|
| head-on (x) | 551 cm^2 | 1.48 | 3.68 N |
| side-on (y) | 422 cm^2 | 1.94 | 3.69 N |
| heave (z) | 692 cm^2 | 1.40 | 4.35 N |

Above 1 because the parts, spread apart, each meet the flow (open-frame
underwater vehicles have 1–1.5 on frontal area). Legs and crown seen side-on are
partly in line, which in reality shields the downstream ones; not modelled. The
file is stamped with the XML's hash, the crown's volume and the targets, and is
refused when any changes.

### 6.3 MuJoCo Warp's fluid torque

MuJoCo Warp 3.11.0 sums each geom's fluid force into its body's wrench and
applies it at the body's centre of mass **without the moment of each geom's force
about that point**; native MuJoCo applies each geom's force at the geom. Forces
agreed, torques did not (kitted robot, 0.3 m/s head current: root pitch moment
−0.0247 N m on Warp, −0.0134 N m native). Since training runs on Warp and replays
on native MuJoCo, [`oceanking/warp_patch.py`](../oceanking/warp_patch.py) replaces
Warp's fluid launcher with a copy of its kernel plus the missing line, installed
when `oceanking.hydro` is imported. It refuses to install unless the installed
MuJoCo Warp's kernel is byte-for-byte the one copied, so an upstream change is
looked at, not overridden. With it, fluid forces and torques agree between the
simulators (test: off-centre geom exactly; the moving kitted robot within 0.2%).

### 6.4 Current

`option.wind`: uniform, constant, horizontal, world frame (`--current VX VY`). Drag
acts on `v_body - v_current`, so a body at rest in a current is pushed downstream
and a body moving with it feels nothing. The value is the water speed **at the
robot's height**; a real near-bed current is slower than the free stream.

`--drag-scale k` multiplies the water's density and viscosity as the drag model
sees them, for sensitivity studies. Buoyancy does not depend on it.

## 7. Added mass

A body accelerating in water accelerates water with it. MuJoCo's fluid models
include only velocity-dependent added-mass terms, so the acceleration term is
added: each link and the crown gain `dm = C_a rho V` of mass, their inertia is
scaled by `(m + dm) / m` (diagonal inertia in the principal frame, so the scaling
is exact), and the extra weight is compensated
(`gravcomp = (gravcomp m + dm) / (m + dm)`). Net weight, buoyancy, contact loads and
the static balance are unchanged; only inertia grows.

| Group | C_a | Why |
|---|---|---|
| torso, crown | 0.5 | compact bodies (a sphere's value) |
| legs, arms | 1.0 | cylinders moving across their axis; legs swing broadside |

`--added-mass K` multiplies all of them (1 by default, 0 off). Applied with drag
(modes `drag` and `full`). For beta 0.8: **3.01 kg** of added mass, 55% of the
5.50 kg robot; the free sink acceleration drops from 1.96 to 1.27 m/s^2, and the
terminal sink speed against heave drag is 0.47 m/s.

**Approximations:** isotropic; at each link's centre of mass rather than its
centre of buoyancy, with its inertia following the link's mass distribution
rather than its shape (largest for the upper shell, whose added mass is 2.7 times
its own). MuJoCo's velocity-dependent added-mass terms are kept as the ellipsoid
model computes them, scaled by each group's multiplier, and partly overlap the
rigid added mass (order 0.1 N while walking).

## 8. The seabed (scenes)

The ground belongs to the **scene** (`--scene`), the water to the task, so each
ground can be run wet and dry. Friction is set on the ground with a contact
priority above the feet's (upstream's `set_ground_contact`), so the ground's
value is the one the solver uses and the feet's dry 1.2 is discarded; the feet's
contact stiffness is kept unless the ground is soft. Upstream's per-environment
foot-friction randomisation therefore has no effect on these grounds.

| Scene | Ground | Sliding friction | Contact | Basis |
|---|---|---|---|---|
| `seabed` | flat, hard | feet's 1.2 (unchanged) | feet's | reference ground, not a claim about any real seabed |
| `seabed_rock` | heightfield 24 x 24 m, 20 mm peak-to-peak (3.8 mm rms), 0.2 m stones, fixed seed, inside a flat 12 m border | **0.5** | feet's | rubber on wet rock: ~0.5–0.7 on rough clean faces, ~0.2–0.4 polished or with biofilm |
| `seabed_sand` | flat, soft (feet sink ~1 cm on land) | **0.6** | upstream `soft`: compliant, plus granular drag on a sunk foot | a foot sliding on saturated sand shears the sand: tan of the internal friction angle, 30–35 deg for clean sand, is 0.58–0.70 |

Neither friction value is a measurement of this robot's feet; both are
literature-range choices and the first parameters to vary in a sensitivity study.
The sand's granular drag is upstream's, which records that its constant has no
measured basis.

**The rock field cannot be left within an episode.** No termination ends an
episode when a robot leaves the ground (upstream drops `out_of_terrain_bounds`).
A training episode is 20 s at up to 0.8 m/s in x and y at once from a spawn within
0.5 m: at most 23.1 m, against the 24 m from the centre to the border's edge
(checked against the task's own numbers by a test). Contact budgets are 320
contacts and 1400 constraint rows per world: the spawn state, root at the rock,
needs 253 and 1034.

With these frictions the traction limit of section 11 becomes, at beta 0.8, a
relative water speed of about **0.36 m/s on rock** and **0.40 m/s on sand**
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
| `--drag-scale K` | `OCEAN_DRAG_SCALE` | 1 | multiplier on drag (water density as the drag model sees it) |
| `--added-mass K` | `OCEAN_ADDED_MASS` | 1 | multiplier on added mass (section 7) |
| `--nominal` | `OCEAN_NOMINAL` | off | replays only (refused in training): start at the origin facing +x, no startup randomisation (foot friction 1.2, no COM offset, no encoder bias) |
| `--trajectory-out PATH` | `OCEAN_TRAJECTORY_OUT` | — | replays only: record root pose and command every control step |

Every effect is a model edit made before compilation: MuJoCo applies it in every
physics substep (1 kHz) on both backends, with nothing to clear on reset; an
independent review confirmed that gravcomp, masses, fluid flags and options
survive startup randomisation and `set_const` in every world on both backends.
`--hydro off` leaves the robot exactly as upstream builds it. Training, replay and
evaluation of one experiment must be given the same values; replays print the
resolved condition and `--trajectory-out` stores it. **Training runs do not yet
record it** (section 13).

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
| Off-centre geom's fluid torque | Warp (patched) equals native; control (unpatched) does not |

On the real Jumper model:

| Check | Result |
|---|---|
| Volume table vs model | SHA-256 match required; masses must agree per link |
| Net weight of the compiled robot | `(1 - beta) M g` (rel. 1e-6) for several `beta`, `a` |
| Each link's buoyancy | `rho V_i g`, at the stored centroid, link itself uncompensated |
| Crown's buoyancy | `rho V_crown g` at the tank's centroid, above the crown's centre of mass |
| Inertial frames of the kit and buoyancy points | as intended (section 12, item 3) |
| Trim | horizontal offset of centre of buoyancy from centre of mass < 1 µm at beta 0.7, 0.8, 0.95 |
| Keel inside the hull | every corner inside the base's hull at beta 0.6, 0.8, 0.95 |
| Standing at HOME under the policy's PD gains | ground load `(1 - beta) M g` (within 0.2% of weight) at beta 0.8 and 0.95 |
| Whole robot, free water, no drag | centre of mass accelerates at `-(1 - beta) g` on both simulators |
| Added mass | sink acceleration `(1 - beta) M g / (M + sum C_a rho V)` on both simulators; inertia grows, net weight does not |
| Rolled 30 deg | restoring moment with the kit; uniform model has none beyond `(1 - beta)` of land |
| Dropped upside down (buoyancy zeroed) | rests on the crown; control without crown collision sinks into the floor |
| Fluid forces, moving kitted robot in a current | Warp (patched) equals native within 0.2%; control (unpatched) does not |
| Drag calibration | stored multipliers reproduced by a fresh measurement; legs' multiplier > 2x torso's; whole-robot drag area = sum of group targets (5%); stale file refused; multipliers reach the model |
| Buoyancy points and keel | keel flagged with negligible multiplier; points' fallback drag under 0.1% of the robot's |
| Seabed grounds | rock and sand outrank the feet with friction 0.5 / 0.6 (rock: feet's stiffness; sand: soft); the reference seabed does not |
| Rock extent | an episode's maximum reach (from the task's own numbers) is inside the ground |
| Through the real task and scene path | `beta`, kit, drag flags, density and current reach the compiled model under any `--scene` |

## 11. Measurements

Workstation: RTX PRO 6000 Blackwell, Ryzen 9 9950X3D; MuJoCo 3.11.0, MuJoCo Warp
3.11.0 (patched, section 6.3).

**Physical quantities** (`physics_report`, beta 0.8, crown air 0.5): weight 53.97 N,
buoyancy 43.17 N, net 10.79 N; BG 9.6 mm; head-on drag 3.68 N at 0.3 m/s; added
mass 3.01 kg; terminal sink 0.47 m/s; traction `mu (1 - beta) M g` = 12.95 N with
the nominal foot friction 1.2, against which the head-on drag balances at a
relative water speed of **0.56 m/s** (beta 0.9: 0.38 m/s; beta 0.95: 0.26 m/s).
Section 8 gives the limit on wet rock and sand.

**Training cost**, `ocean.posture`, 4096 environments, flat ground, per PPO
iteration (2026-10-25): land 1.31 s; water 1.88 s. `seabed_rock`, 1024
environments: 1.7 s.

**Righting** (2026-10-25, before the second review's changes; native, beta 0.8,
joints PD-held at HOME, no policy): dropped upside down from 0.35 m, the robot lands
on its crown and the kit's moment turns it from inverted (body z-axis . world z =
−1.00) to 0.93 (21 deg from upright) within 2.5 s.

**Official `jumper.posture` policy on each ground** (2026-10-25, after both reviews,
`--nominal`, one run each, native backend, same command script: forward 0.5 m/s,
sideways 0.3 m/s, turn, then stand; "hold" is the last 4 s, commanded to stand
still; m/s):

| Condition | forward (cmd 0.5) | sideways (cmd 0.3) | hold | final position |
|---|---|---|---|---|
| Land, flat | 0.42 | 0.22 | 0.00 | (1.59, 0.73) m |
| Land, rock ground | 0.45 | 0.23 | 0.00 | (1.73, 0.81) m |
| Land, sand ground | 0.36 | 0.19 | 0.00 | (1.33, 0.64) m |
| Water beta 0.8, flat | 0.29 | 0.21 | 0.00 | (1.03, 0.66) m |
| Water beta 0.8, rock | 0.23 | 0.20 | −0.01 | (0.80, 0.62) m |
| Water beta 0.8, sand | 0.21 | 0.18 | −0.01 | (0.70, 0.51) m |
| Water, flat, head current 0.3 | 0.11 | 0.20 | −0.06 | (−0.17, 0.61) m |
| Water, rock, head current 0.2 | 0.09 | 0.18 | −0.06 | (−0.20, 0.48) m |
| Water, rock, head current 0.3 | **0.02** | 0.16 | **−0.09** | **(−0.75, 0.43) m** |
| Water, sand, head current 0.3 | **0.02** | 0.16 | **−0.10** | **(−0.88, 0.45) m** |
| Water, rock, cross current 0.3 | 0.22 (drift −0.15) | **0.02** | −0.09 | (0.76, −1.04) m |

Repeat runs of the same nominal condition agree to about 2 cm in final position.
It never fell (body z-axis never below 0.99, no resets). It **fails by being
carried away**: against a head current of 0.2–0.3 m/s it ends behind its start,
and it cannot stand still in a current, losing 0.06–0.10 m/s downstream while
commanded to hold. Standing on all six feet would hold: on rock at beta 0.8 the
head-on drag at 0.3 m/s is 3.7 N against 5.4 N of traction. The policy keeps
stepping in place, and each lifted foot gives up its share. Not yet separated:
how much of the still-water slow-down is drag, reduced traction, or the kit's mass
and added mass.

## 12. Reviews of 2026-10-25

**First review** (gravity, mass and buoyancy re-derived; drag against an
independent estimate):

1. **Drag was about three times too high** (Cd 3.1 head-on): calibrated (section
   6.2). The land-vs-water slow-down measured before was exaggerated by it.
2. **Centre of buoyancy at each link's centre of mass** overstated stability:
   buoyancy now acts at the centroids (section 4).
3. **Explicit inertials left `ipos` unset.** MjSpec's default is NaN and the
   compiler then sets the centre of mass to the body's position again, doubling
   the offset: the keel's centre of mass sat twice as far from the torso origin
   as intended. Every added body now sets `ipos`, and a test pins it.
4. **No trim**: the centre of buoyancy lay 10 mm ahead of the centre of mass, a
   standing 0.43 N m nose-up moment. The keel is now placed to cancel it.
5. **Added mass was missing** (section 7).

**Second review** (three independent reviewers: buoyancy and kit; drag and
simulator consistency; integration with training):

1. **Fatal: robots could walk off the rock field.** An 8 m tile with no border, no
   termination for leaving the ground, and every environment on one patch of
   rock. Now 24 m with a 12 m border, checked against the task's reach (section 8).
2. **The rock scene crashed on MuJoCo Warp** (contact budgets copied from upstream's
   rough scene). Raised (section 8).
3. **MuJoCo Warp's fluid torques lacked each geom's lever arm**: patched (6.3).
4. **One drag factor cut the legs to half a cylinder's drag**: per-group
   calibration (6.2).
5. Smaller, fixed: per-group added-mass coefficients (limbs 1.0); keel footprint
   and height so it lies inside the hull; the crown's water settled at the tank
   floor and its buoyancy kept at the tank's centroid (moving the centre of mass
   alone had moved the buoyancy with it); the keel plate kept by the native
   backend's stripper and given no fluid force; `--nominal` refused in training;
   drag flags restricted to the robot.

## 13. Known limitations and open items

1. The simulated robot is 2.543 kg; the product is 1.8 kg (section 3.1).
2. Volumes are convex hulls of collision meshes, every link sealed; 11% below the
   visual hulls (section 3.2). The table's stamp covers the XML, not the meshes.
3. Drag: per-group factors on MuJoCo's per-geom model; no shielding between parts;
   no lift (section 6).
4. Added mass isotropic, at link centres of mass (section 7); the crown tank's
   free surface is not modelled (section 5).
5. The kit is welded; the crown does not collide with the robot's own limbs.
6. Fully submerged only; no water surface; current uniform down to the seabed.
7. Friction on rock (0.5) and sand (0.6) are literature-range choices (section 8).
8. **For the training design** (physics is right; the task's settings are not yet
   adapted): upstream's foot-load rewards saturate at 5 N and 3 N per foot, set for
   the land weight, which the underwater net weight cannot reach; foot-friction
   randomisation is inert on the rock and sand grounds; upstream's COM-offset
   randomisation (±25/±25/±30 mm on the base) was sized for land and weighs about
   twice as much underwater; training runs do not record the water condition
   or the scene.
