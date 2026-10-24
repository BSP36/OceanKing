# OceanKing

**A tiny crab. A big ocean.** Jumper, a 22-DoF crab robot, learns to walk on the seabed in
simulation despite buoyancy and currents.

## Layout

| Path | Contents |
|---|---|
| `jumper/` | The upstream [Jumper](https://github.com/KingKongRobotics/jumper) repository (Apache-2.0), merged with its history. Training, replay, assets. |
| `oceanking/` | OceanKing's own code: `hydro.py` (underwater physics), `tasks/` and `scenes/`. |
| `tests/` | OceanKing's tests. |
| `docs/` | OceanKing's own documents. |
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

`oceanking/hydro.py`, applied to the model before compilation so it acts in every physics
substep on both backends:

- **Buoyancy**: `body.gravcomp = beta` on every body, i.e. a buoyant force `beta * m * g` at
  each body's centre of mass. Masses, inertias and gravity are unchanged.
- **Drag and current**: MuJoCo's inertia-box fluid model (`option.density`, `viscosity`)
  with `option.wind` as the current, so drag acts on the velocity relative to the water.
- **Depth** changes only the light in the `seabed` scene (`--depth`); fully submerged rigid
  bodies feel no depth dependence in this model.

`ocean.posture` is upstream `jumper.posture` with this water installed. Its observation and
action are unchanged, so the official `jumper.posture` checkpoint runs in it directly:

```bash
cd jumper
CKPT=tasks/jumper/posture/out/example/model_74800.pt
../.venv/bin/python scripts/play.py --task ocean.posture --checkpoint $CKPT --scene seabed \
    --hydro full --buoyancy 0.5 --current 0.0 0.1                  # live viewer
MUJOCO_GL=egl ../.venv/bin/python scripts/play.py --task ocean.posture --checkpoint $CKPT \
    --scene seabed --hydro off --backend native --device cpu --headless \
    --steps 2000 --video land.mp4                                   # the land baseline
```

| Option | Meaning |
|---|---|
| `--hydro {off,buoyancy,drag,full}` | which effects are on (`off` is land) |
| `--buoyancy BETA` | buoyancy-to-weight ratio, 0 land to 1 neutral (default 0.5) |
| `--current VX VY` | water velocity in the world frame, m/s |
| `--drag-scale K` | multiplier on the drag model's density |
| `--depth M` | (`--scene seabed`) water depth for the light |

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
