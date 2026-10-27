# OceanKing grasping under water (ocean.five_foot)

**Status (2026-10-27): verified in still water at beta 0.8.** The official `jumper.five_foot`
policy (left-front arm carried as a claw, five legs walking) runs under water unchanged
(`ocean.five_foot`), and with two arm presets found here the claw picks up, carries and releases
each piece of the demo's sunken trash.

## Walking with the claw under water

The official five-foot checkpoint (`five_foot/out/example/model_86600.pt`), in `ocean.five_foot`
at beta 0.8, still water (native, 2026-10-27): a 0.2 m/s forward command walks 0.15-0.16 m/s, a
0.8 rad/s turn turns at 0.80 rad/s, upright throughout (min body-up 0.987).

## The props under water (`oceanking/props.py`)

MuJoCo's fluid model gives drag and no buoyancy (a density-2000 box at rest in density-1000
water accelerates at the full -9.81 m/s^2), so every prop gets Archimedes' buoyancy as gravity
compensation, `rho V / m`, with `V` its own collision volume (the prop compiled again at the
water's density). Upstream's row under water: soap 0.93 (sinks), notebook 2.45 and the sealed
can 1.07 (float). The demo's trash (`oceanking/demo/scene.py`) sinks, with effective densities
1300-1500 kg/m^3 (flooded can, flooded glass bottle, waterlogged carton): gravcomp 0.67-0.77.

## Arm presets

The task's own hold (`claw.py::LF_GRASP`) is a stow: its jaws close front to back and grasp
nothing. Two presets were searched with `five_foot/tools/grasp_pose.py`'s kinematics (mouth
position, closing axis, aperture; the full clearance measure costs 1.3 s per pose and the
upstream search would take ~6 h, so the search used the mouth alone and the result was measured
in full once):

| preset | J0, J1, J2, J3 (rad) | mouth |
|---|---|---|
| `reach` | -1.1013, -1.3142, -0.8291, 0.6949 | ahead of the left shoulder, jaws horizontal, opening left-right, tips ~8 cm off the floor, 75 mm apart |
| `lift` | -1.1013, -2.0142, -0.8291, 0.6949 | the same with the shoulder pitched 0.7 rad: raises a gripped object ~5 cm |

Measured at `reach` (the original mouth target): lift 57 mm/rad, lowest arm point 60 mm off the
floor, 22 mm from the trunk.

**The mouth is between the jaw tips.** The kinematic estimate of the mouth centre landed on one
jaw tip in the simulation, so `OceanSim.mouth()` is 25 mm inside the midpoint of the two tips,
towards the hinge. The 66 mm can first tried did not fit (75 mm at the tips, 58 mm 55 mm in): the
trash is 48-52 mm across.

## The check (`oceanking/tools/grasp_check.py`)

For each piece: arm at `reach`, claw open, the trash on the floor at the mouth; close; `lift`
(grasped: it rises at least half of the mouth's 4 cm); walk 0.1 m/s while turning 0.3 rad/s for 4 s
(carried: under 4 cm of slip relative to the mouth, still off the floor); open (released: back on
the floor). Control: the can 10 cm to the side must not come up.

| condition | can | bottle | carton | control |
|---|---|---|---|---|
| beta 0.8, still water | rise 55 mm, slip 1 mm, released | 48 mm, 0.4 mm, released | 50 mm, 9 mm, released | not grasped |
| beta 0.8, current 0.1 m/s along x | fell over before the grip | grasped, carried | fell over | not grasped |
| beta 0.8, current 0.1 m/s along y | grasped, carried | fell over | fell over | not grasped |
| beta 0.9, still water | grasped, slipped 96 mm while walking | grasped, carried | grasped, slipped 85 mm | not grasped |

**Passes in still water at beta 0.8** (the demo's condition). In a current the robot itself is
carried a few centimetres while the claw closes and knocks the trash over (the current's drag on
the trash alone, ~0.03 N, is far below what tips it); at beta 0.9 the lighter robot's walk shakes
two of the three out of the claw. Both are limits of grasping with a land policy that does not
know it holds anything, not of the claw.
