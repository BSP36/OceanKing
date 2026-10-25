# OceanKing

**A tiny crab. A big ocean.** Jumper, a 22-DoF crab robot, learns to walk on the seabed in
simulation despite buoyancy and currents.

## Layout

| Path | Contents |
|---|---|
| `jumper/` | The upstream [Jumper](https://github.com/KingKongRobotics/jumper) repository (Apache-2.0), merged with its history. Training, replay, assets. |
| `oceanking/` | OceanKing's own code: `hydro.py`, `buoyancy.py` (underwater physics), `trajectory.py`, `tasks/`, `scenes/`, `tools/`, `data/`. |
| `tests/` | OceanKing's tests. |
| `docs/` | OceanKing's own documents; [`PHYSICS.md`](docs/PHYSICS.md) is the physics specification. |
| `scratch/` | Disposable experiments and outputs (git-ignored; may be deleted at any time). |

### How OceanKing plugs into Jumper

Upstream's `scripts/train.py`, `play.py` and `export.py` are used unchanged. A task id is a
module under `jumper/tasks/` and a scene id a module under `jumper/scenes/`, so OceanKing's
sources live in `oceanking/` and `jumper/` holds symlinks to them plus one line marked
`[oceanking]` in each package's `__init__.py`:

```
jumper/tasks/ocean        -> oceanking/tasks/ocean        (task ids ocean.*)
jumper/scenes/seabed.py   -> oceanking/scenes/seabed.py   (scene id seabed)
```

Physics that defines the experiment (water) belongs to a **task**; what the world looks like
and what the ground is (seabed, rock, sand, seaweed, mission objects) belongs to a **scene**.
`oceanking/plugin.py` explains how to add either.

Upstream changes are merged into `jumper/` with:

```bash
git fetch upstream
git merge -X subtree=jumper upstream/main
```

## Underwater physics

**Specification: [`docs/PHYSICS.md`](docs/PHYSICS.md)** — conventions, the robot's mass and
volume, buoyancy, the ballast kit, drag and current, options, verification and measurements.

In short: each link displaces its measured hull volume at its own centre of buoyancy (sealed,
Jumper floats at beta 1.35); a lead keel inside the torso, sized and placed to trim, and a buoyant
crown tank on its roof bring the whole robot to the requested beta; drag is MuJoCo's ellipsoid
fluid model, calibrated part by part (torso, crown, legs, arms); added mass is included; the
current is `option.wind`. MuJoCo Warp's fluid torque is patched to match native MuJoCo
(`oceanking/warp_patch.py`). `python -m oceanking.tools.physics_report` prints the quantities.

`ocean.posture` is upstream `jumper.posture` with this physics; its observation and action are
unchanged, so the official checkpoint runs in it directly:

```bash
cd jumper
CKPT=tasks/jumper/posture/out/example/model_74800.pt
../.venv/bin/python scripts/play.py --task ocean.posture --checkpoint $CKPT --scene seabed \
    --buoyancy 0.8 --current -0.3 0                                  # live viewer
MUJOCO_GL=egl ../.venv/bin/python scripts/play.py --task ocean.posture --checkpoint $CKPT \
    --scene seabed --hydro off --nominal --backend native --device cpu --headless --steps 2000 \
    --trajectory-out land --video land.mp4 --no-video-follow --video-azimuth 90 \
    --video-distance 4 --video-elevation -30 --video-lookat-height 0   # land baseline, fixed camera
```

Scenes (`--scene`): `seabed` (flat, hard reference ground), `seabed_rock` (wet rock, uneven,
friction 0.5), `seabed_sand` (soft sand, friction 0.6); the water is the task's, so each ground
runs wet or dry. Comparison videos with distance read-outs: `oceanking/tools/compare_video.py`. The volume
table is regenerated with `.venv/bin/python -m oceanking.tools.link_volumes`.

## Setup

Tested on Linux with an RTX PRO 6000 Blackwell (compute capability 12.0), Python 3.12.

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
VIRTUAL_ENV=.venv uv pip install -e "./jumper[dev]" -e ".[dev]"
```

Tests:

```bash
.venv/bin/python -m pytest tests/ -q                       # OceanKing
cd jumper && ../.venv/bin/python -m pytest tests/ -q       # upstream, including OceanKing's tasks
```

The Rust controller toolchain (upstream `docs/AGENT_SETUP.md` sections 2.1–2.2) is only needed for
deploying to the physical robot and is not installed here.
