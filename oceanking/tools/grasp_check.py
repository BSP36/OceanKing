"""Can the claw pick up the demo's trash under water? End to end, measured.

Upstream's `five_foot/tools/grasp_objects.py` asks this on land for its row; this
asks it on the seabed, of `oceanking.demo.scene.TRASH`, in `OceanSim` (the official
five-foot policy standing, the water of `ocean.five_foot`, each prop with its
buoyancy and drag). For each piece of trash:

  1. the arm goes to `reach` with the claw open, and the trash is put on the floor
     with its axis through the mouth;
  2. the claw closes; **grasped**: raising the arm to `lift` raises the trash at
     least `FOLLOWED` of the mouth's rise;
  3. the robot walks forward at 0.1 m/s and turns for 4 s; **carried**: it is
     still in the claw (within `CARRY_SLIP` of where it was relative to the mouth);
  4. the claw opens; **released**: it is back on the floor within 2 s.

Then a control: the same trash put 10 cm to the side of the mouth must **not**
come up when the claw closes and the arm lifts.

    .venv/bin/python -m oceanking.tools.grasp_check [--beta 0.8] [--current VX VY]

The arm presets are measured by `five_foot/tools/grasp_pose.py`'s kinematics
(docs/GRASP.md): `reach` puts the open mouth 0.28 m ahead and 0.11 m left of the
base, 67 mm off the floor, the jaws closing across (left-right); `lift` is the
same with the shoulder raised 0.7 rad, the mouth 106 mm off the floor.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

#: (J0, J1, J2, J3), radians; see the module docstring and docs/GRASP.md.
GRASP_PRESETS = {
    "reach": (-1.1013, -1.3142, -0.8291, 0.6949),
    "lift": (-1.1013, -2.0142, -0.8291, 0.6949),
}
#: The mouth centre in the base frame at `reach`, from `grasp_pose.Claw` at STAND_Z
#: (x ahead, y left, z up from the base origin), m.
MOUTH_AT_REACH = (0.277, 0.115, -0.039)
FOLLOWED = 0.5
CARRY_SLIP = 0.04


def _rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mouth_world(sim) -> np.ndarray:
    return sim.mouth()


def place(sim, name: str, xy: np.ndarray) -> None:
    import torch

    from oceanking.demo.scene import TRASH

    t = next(t for t in TRASH if t.name == name)
    ent = sim.env.scene[name]
    state = torch.zeros(1, 13)
    o = sim.env.scene.env_origins[0]
    state[0, 0], state[0, 1] = float(xy[0]) + float(o[0]), float(xy[1]) + float(o[1])
    state[0, 2] = t.half_height + 0.003 + float(o[2])
    state[0, 3] = 1.0
    ent.write_root_state_to_sim(state)


def park_all(sim) -> None:
    for k, name in enumerate(sim.trash):
        place(sim, name, np.array([3.0 + k, 3.0]))


def trial(sim, name: str, offset: float = 0.0) -> dict:
    sim.claw(closed=False, settle=0.2)
    sim.arm("reach", settle=0.6)
    park_all(sim)
    m = mouth_world(sim)
    place(sim, name, m[:2] + np.array([0.0, offset]))
    sim.stand(0.6)
    z0 = sim.positions()[name][2]
    sim.claw(closed=True, settle=1.0)
    m0 = mouth_world(sim)
    sim.arm("lift", settle=0.8)
    rise_mouth = 0.04  # 106 - 67 mm, from the kinematics
    rise = sim.positions()[name][2] - z0
    grasped = rise >= FOLLOWED * rise_mouth
    rel0 = sim.positions()[name] - mouth_world(sim)
    sim.walk(0.1, 0.0, 0.3, 4.0)
    rel1 = sim.positions()[name] - mouth_world(sim)
    slip = float(np.linalg.norm(rel1 - rel0))
    carried = grasped and slip < CARRY_SLIP and sim.positions()[name][2] > z0 + 0.01
    sim.claw(closed=False, settle=2.0)
    released = bool(sim.positions()[name][2] < z0 + 0.01)
    sim.arm("reach", settle=0.3)
    del m0
    return {"trash": name, "offset_m": offset, "rise_m": round(float(rise), 4),
            "grasped": bool(grasped), "slip_m": round(slip, 4), "carried": bool(carried),
            "released": released}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--current", type=float, nargs=2, default=(0.0, 0.0))
    ap.add_argument("--out", default=None, help="write the results as JSON here")
    args = ap.parse_args()
    from oceanking.demo.sim import OceanSim

    sim = OceanSim(beta=args.beta, current=tuple(args.current), presets=GRASP_PRESETS)
    results = [trial(sim, name) for name in sim.trash]
    results.append(trial(sim, sim.trash[0], offset=0.10))  # the control
    sim.close()
    for r in results:
        print(r)
    ok = all(r["grasped"] and r["carried"] and r["released"] for r in results[:-1])
    ok = ok and not results[-1]["grasped"]
    print("PASS" if ok else "FAIL")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"beta": args.beta, "current": list(args.current), "results": results,
                       "pass": ok}, f, indent=2)


if __name__ == "__main__":
    main()
