# ActiveView: Hybrid View Selection for Efficient 3D Reconstruction

CS184/284A Spring 2026 — UC Berkeley. Full proposal: [`index.html`](index.html).

**Team:** Alexander Gasca Rosas · Winfred Wang · Dennis Liang · Yousif Yacoby

**Status:** recon backend wired (Open3D TSDF + raycast renderer). View-selection loop next.

## Setup

Python 3.10+. Work from repo root.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bat
python -m venv .venv
.venv\Scripts\activate          :: PowerShell or cmd
pip install -r requirements.txt
```

Git Bash: `source .venv/Scripts/activate`.

**OneDrive workaround:** if repo lives under OneDrive, full Open3D install can hit Windows path-length limits. Two-step install dodges it:

```bash
pip install -r requirements.txt
pip install "open3d>=0.18.0" --no-deps
```

Open3D 0.18+ ships prebuilt wheels for arm64 mac, linux, windows. No CUDA needed.

## Smoke test

Loads bunny → renders 12 RGB-D views → fuses with TSDF → extracts mesh → scores vs ground truth.

```bash
python smoke_test.py
```

Drops in `output/`:

- `recon.ply`    — recon mesh
- `gt_mesh.ply`  — normalized ground-truth mesh
- `cameras.npz`  — intrinsics + N extrinsics used
- `views.png`    — RGB + depth montage

Prints bidirectional Chamfer vs ground truth.

## Visualize

Open 3D viewer with recon + GT + camera frustums:

```bash
python visualize.py
```

Mouse = orbit. Scroll = zoom. `Q`/`Esc` = quit. Orange = recon. Gray = GT (offset right). Red = cameras.

## Mesh dataset (`data/`)

Stanford mesh download + `MeshDataset` + preview tools live in [`data/`](data/). See [`data/README.md`](data/README.md) for usage.

## View proposal

```bash
open index.html             # macOS
start index.html            # Windows
xdg-open index.html         # Linux
```

Local server: `python -m http.server 8000` → http://localhost:8000/index.html

## Files

| Item               | Job                                                |
| ------------------ | -------------------------------------------------- |
| `index.html`       | Project proposal                                   |
| `data/`            | Mesh dataset helpers (see `data/README.md`)        |
| `recon.py`         | `Reconstructor` wrap Open3D `ScalableTSDFVolume`   |
| `render.py`        | RGB-D via `RaycastingScene`. No GL context needed  |
| `smoke_test.py`    | End-to-end demo. Dumps to `output/`                |
| `visualize.py`     | Interactive viewer for smoke-test outputs          |
| `requirements.txt` | Python deps                                        |

## License

TBD.
