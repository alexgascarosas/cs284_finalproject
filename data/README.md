# Mesh data for NBV experiments

## Python dependencies

From the **repository root** (see [main README: Python setup](../README.md#python-setup) for macOS vs Windows venv activation):

```bash
pip install -r requirements.txt
pip install "open3d>=0.18.0" --no-deps
```

Use the same `python` as your venv for the commands below (or `.venv/bin/python` on macOS/Linux and `.venv\Scripts\python.exe` on Windows if you skip `activate`).

## Layout

- `meshes/` — Stanford fallback `.ply` files (download script below).
- `shapenet/` — Placeholder for local ShapeNetCore.v2 content (not committed).

## Fallback meshes (no registration)

From the repo root:

```bash
python data/download_fallback.py
```

This downloads the **Stanford Bunny**, **Dragon**, and **Armadillo** from the [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/) into `data/meshes/` as `bunny.ply`, `dragon.ply`, and `armadillo.ply`, printing byte progress for each HTTP download.

## ShapeNet (after access is approved)

1. Obtain **ShapeNetCore.v2** from ShapeNet and unzip it.
2. Point `MeshDataset` / `preview.py` at the folder that **directly contains synset ID directories** (e.g. `03001627`, `02691156`). That folder is usually named `ShapeNetCore.v2` after unzip.

Expected layout per model:

```text
{root}/{synset_id}/{model_id}/models/model_normalized.obj
```

Example: if you unzip into `data/shapenet/ShapeNetCore.v2/`, pass `root=data/shapenet/ShapeNetCore.v2` (adjust to match your actual path).

## Preview

```bash
python data/preview.py --source fallback --index 0
python data/preview.py --source shapenet --root path/to/ShapeNetCore.v2 --category 03001627 --index 0
```

## Useful ShapeNet synset IDs

| Object   | Synset ID  |
|----------|------------|
| Mug      | 03797390   |
| Chair    | 03001627   |
| Airplane | 02691156   |
| Table    | 04379243   |
| Car      | 02958343   |

## Python API

See `dataset.py`: `MeshDataset(source="fallback")` or `MeshDataset(source="shapenet", root=..., category=...)`. Each item is an Open3D `TriangleMesh` centered and scaled to fit inside the **unit sphere** (max vertex distance from origin ≤ 1).
