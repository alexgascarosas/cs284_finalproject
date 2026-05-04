# ActiveView: Hybrid View Selection for Efficient 3D Reconstruction

CS184/284A Spring 2026 - UC Berkeley. Full proposal: [`index.html`](index.html).

**Team:** Alexander Gasca Rosas - Winfred Wang - Dennis Liang - Yousif Yacoby

**Status:** synthetic RGB-D rendering, TSDF fusion, and next-best-view experiments are wired end-to-end.

## Project goal

This project studies **active 3D reconstruction**: if you can choose where to
place the next camera while scanning an object, can you reconstruct the object
faster and more accurately than with random views?

The current codebase simulates that process on known meshes:

1. Start from a ground-truth mesh such as `bunny`, `dragon`, or `armadillo`.
2. Place candidate cameras on a sphere around the object.
3. Render synthetic RGB-D images from those cameras.
4. Score which GT surface samples are visible from each camera.
5. Choose views with a strategy such as `random`, `coverage`, `uncertainty`, or `hybrid`.
6. Fuse the selected views into a TSDF mesh reconstruction.
7. Measure both surface coverage and geometric accuracy over time.

Two metrics are intentionally kept separate:

- **Coverage metrics** track how much of the GT surface has been observed.
- **Geometry metrics** track the quality of the fused TSDF reconstruction mesh.

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
.venv\Scripts\activate
pip install -r requirements.txt
```

Git Bash: `source .venv/Scripts/activate`.

**OneDrive workaround:** if the repo lives under OneDrive, full Open3D install
can hit Windows path-length limits. This two-step install avoids that:

```bash
pip install -r requirements.txt
pip install "open3d>=0.18.0" --no-deps
```

Open3D 0.18+ ships prebuilt wheels for arm64 macOS, Linux, and Windows. No CUDA
is required.

## Smoke test

Loads bunny, renders 12 RGB-D views, fuses them with TSDF, extracts a mesh, and
scores it against ground truth.

```bash
python smoke_test.py
python smoke_test.py --out-dir output_smoke_alt --radius 2.2
```

Outputs:

- `recon.ply` - reconstructed mesh
- `gt_mesh.ply` - normalized ground-truth mesh
- `cameras.npz` - intrinsics plus the extrinsics used
- `views.png` - RGB and depth montage

The script prints a bidirectional Chamfer score against the GT mesh.

## Visualize

Open the Open3D viewer with the default smoke-test outputs:

```bash
python visualize.py
```

Mouse orbits, scroll zooms, and `Q` or `Esc` quits. Orange is the
reconstruction, gray is GT (offset to the right), and red are the cameras.

## Next-best-view experiments

Run one strategy on one mesh:

```bash
python nbv_loop.py --strategy coverage --mesh bunny --run-name bunny_coverage
```

Run all strategies on one mesh:

```bash
python nbv_loop.py --strategy all --mesh bunny --run-name bunny_all
```

Useful tuning example:

```bash
python nbv_loop.py \
  --strategy coverage \
  --mesh armadillo \
  --radius 2.2 \
  --depth-tol 0.02 \
  --visibility-patch-radius 1 \
  --steps 12 \
  --num-candidates 40 \
  --save-steps 1,3,5,10,12 \
  --out-root output_nbv \
  --run-name armadillo_tuned
```

Final experiment settings:

```bash
.venv/bin/python nbv_loop.py \
  --strategy all \
  --mesh bunny \
  --steps 12 \
  --num-candidates 40 \
  --radius 2.2 \
  --width 320 \
  --height 240 \
  --fov 55 \
  --gt-samples 15000 \
  --depth-tol 0.02 \
  --visibility-patch-radius 1 \
  --seed 7 \
  --save-steps 1,3,5,10,12 \
  --out-root output_nbv_final \
  --run-name bunny_final
```

Repeat with:

```bash
.venv/bin/python nbv_loop.py \
  --strategy all \
  --mesh armadillo \
  --steps 12 \
  --num-candidates 40 \
  --radius 2.2 \
  --width 320 \
  --height 240 \
  --fov 55 \
  --gt-samples 15000 \
  --depth-tol 0.02 \
  --visibility-patch-radius 1 \
  --seed 7 \
  --save-steps 1,3,5,10,12 \
  --out-root output_nbv_final \
  --run-name armadillo_final

.venv/bin/python nbv_loop.py \
  --strategy all \
  --mesh dragon \
  --steps 12 \
  --num-candidates 40 \
  --radius 2.2 \
  --width 320 \
  --height 240 \
  --fov 55 \
  --gt-samples 15000 \
  --depth-tol 0.02 \
  --visibility-patch-radius 1 \
  --seed 7 \
  --save-steps 1,3,5,10,12 \
  --out-root output_nbv_final \
  --run-name dragon_final
```

Final plots should compare `mesh_chamfer` and `seen_fraction` over steps for
`random`, `coverage`, `uncertainty`, and `hybrid`.

### Strategies

- `random` - baseline; picks an unused view uniformly at random.
- `coverage` - prefers views that expose the most currently unseen GT samples.
- `uncertainty` - prefers views that revisit already-seen surface with low observation count.
- `hybrid` - weighted coverage + uncertainty policy; keeps exploring unseen surface while gradually increasing refinement weight.

### Output layout

Each run writes to its own folder under `output_nbv/` (or your chosen
`--out-root`):

- `{run_name}/{strategy}/recon.ply` - final TSDF-fused mesh
- `{run_name}/{strategy}/recon_step_XX.ply` - checkpoint meshes
- `{run_name}/{strategy}/gt_mesh.ply` - normalized GT mesh
- `{run_name}/{strategy}/views.png` - montage of selected RGB-D frames
- `{run_name}/{strategy}/cameras.npz` - selected camera extrinsics and intrinsics
- `{run_name}/{strategy}/metrics.csv` - per-step coverage and geometry metrics

Key `metrics.csv` columns:

- `seen_fraction` - fraction of GT samples seen at least once
- `newly_seen_samples` - new GT samples added by that step
- `coverage_chamfer` - Chamfer proxy from visible GT samples only
- `mesh_chamfer` - Chamfer of the actual TSDF mesh
- `mesh_vertices`, `mesh_triangles` - mesh size over time

## Mesh dataset (`data/`)

Stanford mesh download plus `MeshDataset` and preview tools live in [`data/`](data/).
See [`data/README.md`](data/README.md) for usage.

## View proposal

```bash
open index.html
start index.html
xdg-open index.html
```

Or serve it locally:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/index.html`.

## Files

| Item               | Job                                              |
| ------------------ | ------------------------------------------------ |
| `index.html`       | Project proposal                                 |
| `data/`            | Mesh dataset helpers                             |
| `recon.py`         | TSDF fusion wrapper                              |
| `render.py`        | RGB-D renderer and shared camera/depth helpers   |
| `smoke_test.py`    | End-to-end TSDF smoke test                       |
| `nbv_loop.py`      | Candidate generation, view selection, metrics    |
| `visualize.py`     | Interactive viewer for smoke-test outputs        |
| `requirements.txt` | Python dependencies                              |

## License

TBD.
