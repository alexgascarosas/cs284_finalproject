# ActiveView

CS184/284A Spring 2026 final project. Proposal in `index.html`.

**Status:** recon backend wired (Open3D TSDF + raycast renderer). View-selection loop next.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open3D 0.18+ has prebuilt wheels for arm64 mac, linux, windows. No CUDA needed.

## Smoke test

Loads bunny → renders 12 RGB-D views → fuses with TSDF → extracts mesh → scores vs ground truth.

```bash
python smoke_test.py
```

Drops in `output/`:

- `recon.ply`    — recon mesh
- `gt_mesh.ply`  — ground-truth mesh, normalized
- `cameras.npz`  — intrinsics + N extrinsics used
- `views.png`    — RGB + depth montage

Prints bidirectional Chamfer vs ground truth.

## Visualize

Open 3D viewer with recon + GT + camera frustums:

```bash
python visualize.py
```

Mouse = orbit. Scroll = zoom. `Q`/`Esc` = quit. Orange = recon. Gray = GT (offset right). Red = cameras.

## Files

| File              | Job                                                |
| ----------------- | -------------------------------------------------- |
| `recon.py`        | `Reconstructor` wrap Open3D `ScalableTSDFVolume`   |
| `render.py`       | RGB-D via `RaycastingScene`. No GL context needed  |
| `smoke_test.py`   | End-to-end demo. Dumps to `output/`                |
| `visualize.py`    | Interactive viewer for smoke-test outputs          |
| `requirements.txt`| Deps                   |

feel free to reorganize files