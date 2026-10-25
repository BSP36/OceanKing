"""MuJoCo Warp 3.11.0's fluid-force kernel with the geom lever arm restored.

**The bug.** In the ellipsoid fluid model, `_fluid_force` sums each geom's force
and torque into its body's wrench, and `support.apply_ft` applies that wrench at
the body's centre of mass (`xipos`). The moment of each geom's force about
`xipos`, ``(geom_xpos - xipos) x F``, is never added. Native MuJoCo applies each
geom's force at the geom. Forces agree between the two; torques do not -- on the
kitted Jumper in a 0.3 m/s head current the pitch moment on the root was
-0.0247 N m on Warp against -0.0134 N m native (found 2026-10-25 by review;
`tests/test_warp_fluid.py` pins the fix).

**The fix** is the one line marked `[oceanking]` below. Everything else is
MuJoCo Warp's code, copied verbatim from `mujoco_warp/_src/passive.py`
(Apache-2.0, Copyright 2025 The Newton Developers), so that Warp can compile it
from a real source file.

`install()` replaces `mujoco_warp._src.passive._fluid` with `fluid`. It refuses to run unless the
installed MuJoCo Warp's original kernel and launcher are byte-for-byte the ones
this was copied from (`SOURCE_SHA256`): an upstream change, including a fix of
this bug, must be looked at rather than silently overridden.
"""

# ruff: noqa: E501

from __future__ import annotations

import hashlib

import warp as wp
from mujoco_warp._src import passive as passive_module
from mujoco_warp._src import support
from mujoco_warp._src.passive import _pow2, _pow4, ellipsoid_max_moment, geom_semiaxes
from mujoco_warp._src.types import MJ_MINVAL, Data, Model

#: sha256 of MuJoCo Warp 3.11.0's `_fluid_force` kernel and `_fluid` launcher.
SOURCE_SHA256 = "e13cd98dbf397cf5f4841749d731b65202d61e854ad860d2a607bf69f3e45a2a"


@wp.kernel
def _fluid_force(
  # Model:
  opt_wind: wp.array[wp.vec3],
  opt_density: wp.array[float],
  opt_viscosity: wp.array[float],
  body_rootid: wp.array[int],
  body_geomnum: wp.array[int],
  body_geomadr: wp.array[int],
  body_mass: wp.array2d[float],
  body_inertia: wp.array2d[wp.vec3],
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_fluid: wp.array2d[float],
  body_fluid_ellipsoid: wp.array[bool],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  ximat_in: wp.array2d[wp.mat33],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  # Out:
  fluid_applied_out: wp.array2d[wp.spatial_vector],
):
  """Computes body-space fluid forces for both inertia-box and ellipsoid models."""
  worldid, bodyid = wp.tid()
  zero_force = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))

  if bodyid == 0:
    fluid_applied_out[worldid, bodyid] = zero_force
    return

  # skip bodies with negligible mass
  mass = body_mass[worldid % body_mass.shape[0], bodyid]
  if mass < MJ_MINVAL:
    fluid_applied_out[worldid, bodyid] = zero_force
    return

  wind = opt_wind[worldid % opt_wind.shape[0]]
  density = opt_density[worldid % opt_density.shape[0]]
  viscosity = opt_viscosity[worldid % opt_viscosity.shape[0]]

  # Body kinematics
  xipos = xipos_in[worldid, bodyid]
  rot = ximat_in[worldid, bodyid]
  rotT = wp.transpose(rot)
  cvel = cvel_in[worldid, bodyid]
  ang_global = wp.spatial_top(cvel)
  lin_global = wp.spatial_bottom(cvel)
  subtree_root = subtree_com_in[worldid, body_rootid[bodyid]]
  lin_com = lin_global - wp.cross(xipos - subtree_root, ang_global)

  if body_fluid_ellipsoid[bodyid]:
    force_global = wp.vec3(0.0)
    torque_global = wp.vec3(0.0)

    start = body_geomadr[bodyid]
    count = body_geomnum[bodyid]

    for i in range(count):
      geomid = start + i
      coef = geom_fluid[geomid, 0]
      if coef <= 0.0:
        continue

      size = geom_size[worldid % geom_size.shape[0], geomid]
      semiaxes = geom_semiaxes(size, geom_type[geomid])
      geom_rot = geom_xmat_in[worldid, geomid]
      geom_rotT = wp.transpose(geom_rot)
      geom_pos = geom_xpos_in[worldid, geomid]

      lin_point = lin_com + wp.cross(ang_global, geom_pos - xipos)

      l_ang = geom_rotT @ ang_global
      l_lin = geom_rotT @ lin_point

      if wind[0] or wind[1] or wind[2]:
        l_lin -= geom_rotT @ wind

      lfrc_torque = wp.vec3(0.0)
      lfrc_force = wp.vec3(0.0)

      if density > 0.0:
        # added-mass forces and torques
        virtual_mass = wp.vec3(geom_fluid[geomid, 6], geom_fluid[geomid, 7], geom_fluid[geomid, 8])
        virtual_inertia = wp.vec3(geom_fluid[geomid, 9], geom_fluid[geomid, 10], geom_fluid[geomid, 11])

        virtual_lin_mom = wp.vec3(
          density * virtual_mass[0] * l_lin[0],
          density * virtual_mass[1] * l_lin[1],
          density * virtual_mass[2] * l_lin[2],
        )
        virtual_ang_mom = wp.vec3(
          density * virtual_inertia[0] * l_ang[0],
          density * virtual_inertia[1] * l_ang[1],
          density * virtual_inertia[2] * l_ang[2],
        )

        added_mass_force = wp.cross(virtual_lin_mom, l_ang)
        added_mass_torque = wp.cross(virtual_lin_mom, l_lin) + wp.cross(virtual_ang_mom, l_ang)

        lfrc_force += added_mass_force
        lfrc_torque += added_mass_torque

      # lift force orthogonal to velocity from Kutta-Joukowski theorem
      magnus_coef = geom_fluid[geomid, 5]
      kutta_coef = geom_fluid[geomid, 4]
      blunt_drag_coef = geom_fluid[geomid, 1]
      slender_drag_coef = geom_fluid[geomid, 2]
      ang_drag_coef = geom_fluid[geomid, 3]

      volume = wp.static(4.0 / 3.0 * wp.pi) * semiaxes[0] * semiaxes[1] * semiaxes[2]
      d_max = wp.max(wp.max(semiaxes[0], semiaxes[1]), semiaxes[2])
      d_min = wp.min(wp.min(semiaxes[0], semiaxes[1]), semiaxes[2])
      d_mid = semiaxes[0] + semiaxes[1] + semiaxes[2] - d_max - d_min
      A_max = wp.pi * d_max * d_mid

      lin_speed = wp.length(l_lin)

      magnus_force = wp.cross(l_ang, l_lin) * (magnus_coef * density * volume)

      s12 = semiaxes[1] * semiaxes[2]
      s20 = semiaxes[2] * semiaxes[0]
      s01 = semiaxes[0] * semiaxes[1]

      proj_denom = _pow4(s12) * _pow2(l_lin[0]) + _pow4(s20) * _pow2(l_lin[1]) + _pow4(s01) * _pow2(l_lin[2])
      proj_num = _pow2(s12 * l_lin[0]) + _pow2(s20 * l_lin[1]) + _pow2(s01 * l_lin[2])

      A_proj = wp.pi * wp.sqrt(proj_denom / wp.max(MJ_MINVAL, proj_num))
      cos_alpha = proj_num / wp.max(MJ_MINVAL, lin_speed * proj_denom)

      norm = wp.vec3(
        _pow2(s12) * l_lin[0],
        _pow2(s20) * l_lin[1],
        _pow2(s01) * l_lin[2],
      )

      kutta_force = wp.vec3(0.0)
      if density > 0.0 and kutta_coef != 0.0 and lin_speed > MJ_MINVAL:
        kutta_circ = wp.cross(norm, l_lin) * (kutta_coef * density * cos_alpha * A_proj)
        kutta_force = wp.cross(kutta_circ, l_lin)

      eq_sphere_D = wp.static(2.0 / 3.0) * (semiaxes[0] + semiaxes[1] + semiaxes[2])
      lin_visc_force_coef = wp.static(3.0 * wp.pi) * eq_sphere_D
      lin_visc_torq_coef = wp.pi * eq_sphere_D * eq_sphere_D * eq_sphere_D

      I_max = wp.static(8.0 / 15.0 * wp.pi) * d_mid * _pow4(d_max)
      II0 = ellipsoid_max_moment(semiaxes, 0)
      II1 = ellipsoid_max_moment(semiaxes, 1)
      II2 = ellipsoid_max_moment(semiaxes, 2)

      mom_visc = wp.vec3(
        l_ang[0] * (ang_drag_coef * II0 + slender_drag_coef * (I_max - II0)),
        l_ang[1] * (ang_drag_coef * II1 + slender_drag_coef * (I_max - II1)),
        l_ang[2] * (ang_drag_coef * II2 + slender_drag_coef * (I_max - II2)),
      )

      drag_lin_coef = viscosity * lin_visc_force_coef + density * lin_speed * (
        A_proj * blunt_drag_coef + slender_drag_coef * (A_max - A_proj)
      )
      drag_ang_coef = viscosity * lin_visc_torq_coef + density * wp.length(mom_visc)

      lfrc_torque -= drag_ang_coef * l_ang
      lfrc_force += magnus_force + kutta_force - drag_lin_coef * l_lin

      lfrc_torque *= coef
      lfrc_force *= coef

      # map force/torque from local to world frame: lfrc -> bfrc
      torque_global += geom_rot @ lfrc_torque
      # [oceanking] the geom's force acts at the geom, but apply_ft applies the
      # body's total at xipos: add its moment about xipos, as native MuJoCo does.
      torque_global += wp.cross(geom_pos - xipos, geom_rot @ lfrc_force)
      force_global += geom_rot @ lfrc_force

    fluid_applied_out[worldid, bodyid] = wp.spatial_vector(force_global, torque_global)
    return

  l_ang = rotT @ ang_global
  l_lin = rotT @ lin_com

  if wind[0] or wind[1] or wind[2]:
    l_lin -= rotT @ wind

  lfrc_torque = wp.vec3(0.0)
  lfrc_force = wp.vec3(0.0)

  has_viscosity = viscosity > 0.0
  has_density = density > 0.0

  if has_viscosity or has_density:
    inertia = body_inertia[worldid % body_inertia.shape[0], bodyid]
    mass = body_mass[worldid % body_mass.shape[0], bodyid]
    scl = 6.0 / mass
    box0 = wp.sqrt(wp.max(MJ_MINVAL, inertia[1] + inertia[2] - inertia[0]) * scl)
    box1 = wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[2] - inertia[1]) * scl)
    box2 = wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[1] - inertia[2]) * scl)

  if has_viscosity:
    diam = (box0 + box1 + box2) / 3.0
    lfrc_torque = -l_ang * wp.pow(diam, 3.0) * wp.pi * viscosity
    lfrc_force = -3.0 * l_lin * diam * wp.pi * viscosity

  if has_density:
    lfrc_force -= wp.vec3(
      0.5 * density * box1 * box2 * wp.abs(l_lin[0]) * l_lin[0],
      0.5 * density * box0 * box2 * wp.abs(l_lin[1]) * l_lin[1],
      0.5 * density * box0 * box1 * wp.abs(l_lin[2]) * l_lin[2],
    )

    scl = density / 64.0
    box0_pow4 = wp.pow(box0, 4.0)
    box1_pow4 = wp.pow(box1, 4.0)
    box2_pow4 = wp.pow(box2, 4.0)
    lfrc_torque -= wp.vec3(
      box0 * (box1_pow4 + box2_pow4) * wp.abs(l_ang[0]) * l_ang[0] * scl,
      box1 * (box0_pow4 + box2_pow4) * wp.abs(l_ang[1]) * l_ang[1] * scl,
      box2 * (box0_pow4 + box1_pow4) * wp.abs(l_ang[2]) * l_ang[2] * scl,
    )

  torque_global = rot @ lfrc_torque
  force_global = rot @ lfrc_force

  fluid_applied_out[worldid, bodyid] = wp.spatial_vector(force_global, torque_global)


def fluid(m: Model, d: Data):
  fluid_applied = wp.empty((d.nworld, m.nbody), dtype=wp.spatial_vector)

  wp.launch(
    _fluid_force,
    dim=(d.nworld, m.nbody),
    inputs=[
      m.opt.wind,
      m.opt.density,
      m.opt.viscosity,
      m.body_rootid,
      m.body_geomnum,
      m.body_geomadr,
      m.body_mass,
      m.body_inertia,
      m.geom_type,
      m.geom_size,
      m.geom_fluid,
      m.body_fluid_ellipsoid,
      d.xipos,
      d.ximat,
      d.geom_xpos,
      d.geom_xmat,
      d.subtree_com,
      d.cvel,
    ],
    outputs=[fluid_applied],
  )

  support.apply_ft(m, d, fluid_applied, d.qfrc_fluid, False)


_installed = False
#: MuJoCo Warp's own launcher, kept for tests to compare against.
original = passive_module._fluid


def _upstream_digest() -> str:
    text = open(passive_module.__file__).read()
    a = text.index("@wp.kernel\ndef _fluid_force(")
    b = text.index("def _fluid(m: Model, d: Data):")
    c = text.index("\n\n\n", b)
    return hashlib.sha256((text[a:b].rstrip() + "\n" + text[b:c] + "\n").encode()).hexdigest()


def install() -> None:
    """Route MuJoCo Warp's fluid forces through the fixed kernel. Idempotent."""
    global _installed
    if _installed:
        return
    if _upstream_digest() != SOURCE_SHA256:
        raise RuntimeError(
            "mujoco_warp's fluid kernel differs from the one oceanking/warp_patch.py "
            "was copied from. Check whether upstream fixed the missing geom lever arm "
            "and re-copy (or drop) the patch."
        )
    passive_module._fluid = fluid
    _installed = True
