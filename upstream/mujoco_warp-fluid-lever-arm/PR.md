# Draft pull request for google-deepmind/mujoco_warp

**Title:** Ellipsoid fluid model: add each geom's moment about the body's center of mass

**Body:**

In `_fluid_force` (ellipsoid branch), each geom's fluid force and torque are summed into the body's wrench,
which `support.apply_ft` then applies at the body's center of mass (`xipos`). The moment of each geom's force
about `xipos`, `(geom_xpos - xipos) x F_geom`, is never added. MuJoCo applies each geom's force at the geom
(`mj_ellipsoidFluidModel` calls `mj_applyFT` with `d->geom_xpos + 3*geomid` as the point), so the two disagree in
torque whenever a fluid geom is not at its body's center of mass. Forces agree.

The geom velocities are already computed at the geom (`lin_point = lin_com + cross(ang_global, geom_pos - xipos)`),
and the fluid derivative (`_deriv_ellipsoid_fluid`) already takes each geom's Jacobian at `geom_pos`, so with
implicit integrators the force and its derivative currently describe different models.

Reproduction (MuJoCo 3.17.0, MuJoCo Warp 3.17.0, `qfrc_fluid` after `forward`, free body, `density="1000"`):

| Model | MuJoCo | MuJoCo Warp |
|---|---|---|
| one fluid geom 0.2 m from the COM, translating at 1 m/s | `[0, -2.3562, 0, 0, 0, -0.4712]` | `[0, -2.3562, 0, 0, 0, -0.0]` |
| two fluid geoms at ±0.2 m, spinning at 3 rad/s | `[0, 0, 0, 0, 0, -0.246]` | `[0, 0, 0, 0, 0, -0.0085]` |
| control: fluid geom at the COM | `[0.0132, -2.3562, 0, 0, 0, -0.0042]` | identical |

The existing `test_ellipsoid_fluid` uses a single geom at the body's origin, where the missing term is zero.

**Change:** add `wp.cross(geom_pos - xipos, geom_force)` to the body torque (`passive.py`, one line), and
`test_ellipsoid_fluid_offset_geoms`: a free body with two rotated, offset fluid geoms and a hinged child, random
velocities and wind, compared with MuJoCo's `qfrc_fluid`. It fails before the change (torque components off by up
to 0.25) and passes after.

**Tests:** `passive_test.py` (51 passed), `derivative_test.py` and `forward_test.py` (288 passed, 10 skipped), the
full suite with `uv run pytest -n 8` (see `README.md` in this directory for the result), `pre-commit` (ruff, ruff
format, kernel-analyzer) passed.
