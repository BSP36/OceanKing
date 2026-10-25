"""Minimal reproduction: MuJoCo Warp's ellipsoid fluid force drops each geom's lever arm.

Compares `qfrc_fluid` between MuJoCo (C) and MuJoCo Warp for three one-body models.
Only `mujoco` and `mujoco_warp` are needed:

    python repro.py

Expected: identical `qfrc_fluid` on both simulators in every case.
"""

import importlib.metadata as md

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

CASES = {
  # One fluid geom 0.2 m from the body's centre of mass, body translating sideways.
  "offset geom, translating": (
    """
    <mujoco>
      <option density="1000" gravity="0 0 0"/>
      <worldbody>
        <body>
          <freejoint/>
          <geom type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/>
          <geom type="box" size=".05 .02 .03" pos=".2 0 0" mass="1e-6" fluidshape="ellipsoid"
                contype="0" conaffinity="0"/>
        </body>
      </worldbody>
    </mujoco>
    """,
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
  ),
  # Two fluid geoms either side of the centre of mass, body spinning about z: the
  # forces cancel, only their moment remains (a paddle wheel's damping).
  "two geoms, spinning": (
    """
    <mujoco>
      <option density="1000" gravity="0 0 0"/>
      <worldbody>
        <body>
          <freejoint/>
          <geom type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/>
          <geom type="box" size=".02 .05 .03" pos=".2 0 0" mass="1e-6" fluidshape="ellipsoid"
                contype="0" conaffinity="0"/>
          <geom type="box" size=".02 .05 .03" pos="-.2 0 0" mass="1e-6" fluidshape="ellipsoid"
                contype="0" conaffinity="0"/>
        </body>
      </worldbody>
    </mujoco>
    """,
    [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
  ),
  # Control: the fluid geom at the centre of mass (what the existing tests cover).
  "geom at centre of mass (control)": (
    """
    <mujoco>
      <option density="1000" gravity="0 0 0"/>
      <worldbody>
        <body>
          <freejoint/>
          <geom type="box" size=".05 .02 .03" mass="1" fluidshape="ellipsoid"/>
        </body>
      </worldbody>
    </mujoco>
    """,
    [0.0, 1.0, 0.0, 0.0, 0.0, 3.0],
  ),
}


def main():
  print(f"mujoco {md.version('mujoco')}, mujoco-warp {md.version('mujoco-warp')}, "
        f"warp {md.version('warp-lang')}")
  wp.init()
  np.set_printoptions(precision=4, suppress=True)
  for name, (xml, qvel) in CASES.items():
    mjm = mujoco.MjModel.from_xml_string(xml)
    mjd = mujoco.MjData(mjm)
    mjd.qvel[:] = qvel
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd)
    mjw.forward(m, d)
    native = mjd.qfrc_fluid
    warp = d.qfrc_fluid.numpy()[0]
    ok = np.allclose(native, warp, rtol=1e-4, atol=1e-6)
    print(f"\n{name}: {'agree' if ok else 'DIFFER'}")
    print(f"  MuJoCo       qfrc_fluid {native}")
    print(f"  MuJoCo Warp  qfrc_fluid {warp}")


if __name__ == "__main__":
  main()
