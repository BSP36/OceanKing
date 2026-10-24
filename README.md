# OceanKing

**A tiny crab. A big ocean.** Jumper, a 22-DoF crab robot, learns to walk on the seabed in
simulation despite buoyancy and currents. See [`docs/PROPOSAL.md`](docs/PROPOSAL.md).

## Layout

| Path | Contents |
|---|---|
| `jumper/` | The upstream [Jumper](https://github.com/KingKongRobotics/jumper) repository (Apache-2.0), merged with its history. Training, tasks and assets live here. |
| `docs/` | OceanKing's own documents. |

Upstream changes are merged into `jumper/` with:

```bash
git fetch upstream
git merge -X subtree=jumper upstream/main
```

## Setup

Tested on Linux with an RTX PRO 6000 Blackwell (compute capability 12.0), Python 3.12.

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
VIRTUAL_ENV=.venv uv pip install -e "./jumper[dev]"
```

Jumper's entry points run from `jumper/`:

```bash
cd jumper
../.venv/bin/python scripts/train.py --list
../.venv/bin/python -m pytest tests/ -q
../.venv/bin/python scripts/train.py --task jumper.flat --num_envs 256 --max-iterations 5 --headless
```

The Rust controller toolchain (upstream `docs/AGENT_SETUP.md` sections 2.1–2.2) is only needed for
deploying to the physical robot and is not installed here.
