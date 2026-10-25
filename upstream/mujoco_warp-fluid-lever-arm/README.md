# MuJoCo Warp: ellipsoid fluid force drops each geom's lever arm

An upstream bug found while building OceanKing's underwater physics, and the material for reporting
and fixing it in [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp).

## The bug

`mujoco_warp/_src/passive.py`, `_fluid_force`, ellipsoid branch: each geom's fluid force and torque
are summed into its body's wrench, and `support.apply_ft` applies that wrench at the body's centre of
mass (`xipos`). The moment of each geom's force about `xipos`, `(geom_xpos - xipos) x F`, is never
added. MuJoCo (C) applies each geom's force at the geom (`mj_ellipsoidFluidModel` ->
`mj_applyFT(..., d->geom_xpos + 3*geomid, ...)`). Forces agree; torques differ whenever a fluid
geom is off its body's centre of mass.

Consequences:

- Rotational fluid damping is far too small for bodies with several fluid geoms (a paddle-like body
  spinning at 3 rad/s: −0.0085 N m on Warp against −0.246 N m in MuJoCo).
- The fluid derivative (`derivative.py`, `_deriv_ellipsoid_fluid`) takes each geom's Jacobian at
  `geom_pos`, i.e. it is correct, so implicit integrators combine a force and a derivative of
  different models.
- Training on Warp and evaluating on MuJoCo runs different hydrodynamics.

The existing upstream test (`test_ellipsoid_fluid`) uses one geom at the body's origin, where the
missing term is zero.

## Status of checks (2026-10-25)

| Check | Result |
|---|---|
| Installed version in OceanKing | 3.11.0: reproduces (`repro_output_3.11.0.txt`) |
| Latest release, PyPI and GitHub | **3.17.0** (2026-10-22): reproduces (`repro_output_3.17.0.txt`) |
| `main` at `2c61420` (2026-10-23) | code unchanged at `passive.py:560-561`; the new test fails |
| Existing issues | none found. Searched `fluid`, `ellipsoid`, `fluid torque`, `fluid force`, `geom_xpos fluid`, `apply_ft`, `lever arm`, `fluidshape`; related but different: #93 (ellipsoid model), #750 (its derivative) |
| Existing PRs | none found. Recent fluid PRs fix other things: #1679 (normalise velocity and semi-axes), #1730 (applied-force projection to subtrees), #1548, #1432, #985 |
| Native reference | MuJoCo `main`, `src/engine/engine_passive.c`, `mj_ellipsoidFluidModel` applies per geom at `geom_xpos` |

## Files

| File | What |
|---|---|
| `repro.py` | standalone reproduction (`mujoco`, `mujoco_warp` only): two failing cases and a control |
| `repro_output_3.11.0.txt`, `repro_output_3.17.0.txt` | its output on each version |
| `fix.patch` | the change against `main` `2c61420`: one line in `passive.py`, one test in `passive_test.py` |
| `PR.md` | draft title and body for the pull request |

The working clone with the patch applied is `scratch/upstream/mujoco_warp` (disposable).

## Upstream tests with the patch

- `passive_test.py`: 51 passed (new test fails without the fix: torque off by up to 0.25).
- `derivative_test.py`, `forward_test.py`: 288 passed, 10 skipped.
- Full suite, `uv run pytest -n 8`: 2248 passed, 28 skipped, 0 failed (7 min 30 s, RTX PRO 6000 Blackwell).
- `pre-commit` on the changed files: ruff, ruff format, kernel-analyzer passed.

## Before opening the PR (the account owner's steps)

1. Sign Google's CLA (https://cla.developers.google.com/) with the GitHub account that will author
   the commit.
2. Fork `google-deepmind/mujoco_warp`, branch from current `main`, apply `fix.patch`, re-run the
   tests above (upstream moves daily).
3. **Commit without AI co-author lines**: upstream's CONTRIBUTING requires every commit author and
   co-author to have signed the CLA and forbids AI assistants as co-authors.
4. Push to the fork and open the PR with `PR.md`.

## In OceanKing

`oceanking/warp_patch.py` applies the same fix to the installed 3.11.0 by replacing Warp's fluid
launcher, and refuses to install if Warp's kernel changes. Once a release with the fix is used, drop
it.
