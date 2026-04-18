"""
Mesh dataset for NBV experiments: fallback .ply in data/meshes/ or ShapeNetCore.v2 layout.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import open3d as o3d

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_FALLBACK_DIR = DATA_DIR / "meshes"
DEFAULT_SHAPENET_ROOT = DATA_DIR / "shapenet"


def normalize_mesh_unit_sphere(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    Center mesh at origin and scale so all vertices lie inside the closed unit ball
    (max vertex norm <= 1). Returns a new mesh (copy).
    """
    m = copy.deepcopy(mesh)
    if len(m.vertices) == 0:
        raise ValueError("Mesh has no vertices; cannot normalize.")
    vertices = np.asarray(m.vertices, dtype=np.float64)
    center = vertices.mean(axis=0)
    vertices = vertices - center
    norms = np.linalg.norm(vertices, axis=1)
    max_r = float(norms.max()) if norms.size else 0.0
    if max_r < 1e-12:
        raise ValueError("Mesh is degenerate (near-zero extent); cannot normalize.")
    vertices = vertices / max_r
    m.vertices = o3d.utility.Vector3dVector(vertices)
    m.compute_vertex_normals()
    return m


def load_mesh_file(path: Path) -> o3d.geometry.TriangleMesh:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Mesh file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix not in (".ply", ".obj"):
        raise ValueError(f"Unsupported mesh format {suffix!r}; use .ply or .obj. Path: {path}")
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise RuntimeError(f"Open3D loaded an empty mesh from {path}.")
    return mesh


class MeshDataset:
    """
    Unified access to fallback PLY meshes or ShapeNet normalized OBJ meshes.

    - source='fallback': all *.ply under meshes_dir (sorted by name).
    - source='shapenet': model_normalized.obj under {root}/{category}/{model_id}/models/
    """

    def __init__(
        self,
        source: Literal["fallback", "shapenet"] = "fallback",
        *,
        root: str | Path | None = None,
        category: str | None = None,
        meshes_dir: str | Path | None = None,
    ) -> None:
        self._source: Literal["fallback", "shapenet"] = source
        self._entries: list[Path] = []
        self._labels: list[str] = []

        if source == "fallback":
            base = Path(meshes_dir) if meshes_dir is not None else DEFAULT_FALLBACK_DIR
            base = base.resolve()
            if not base.is_dir():
                raise FileNotFoundError(
                    f"Fallback mesh directory does not exist: {base}\n"
                    "Create it or run: python data/download_fallback.py"
                )
            paths = sorted(base.glob("*.ply"))
            if not paths:
                raise FileNotFoundError(
                    f"No .ply files found in {base}.\nRun: python data/download_fallback.py"
                )
            self._entries = paths
            self._labels = [p.stem for p in paths]
        elif source == "shapenet":
            if category is None or not str(category).strip():
                raise ValueError(
                    "MeshDataset(source='shapenet') requires a non-empty category (synset_id)."
                )
            root_path = Path(root) if root is not None else DEFAULT_SHAPENET_ROOT
            root_path = root_path.resolve()
            cat_dir = root_path / str(category)
            if not cat_dir.is_dir():
                raise FileNotFoundError(
                    f"ShapeNet category directory not found: {cat_dir}\n"
                    "Point root at your unzipped ShapeNetCore.v2 folder (contains synset ID subfolders)."
                )
            model_dirs = sorted(d for d in cat_dir.iterdir() if d.is_dir())
            obj_paths: list[Path] = []
            labels: list[str] = []
            for md in model_dirs:
                obj = md / "models" / "model_normalized.obj"
                if obj.is_file():
                    obj_paths.append(obj)
                    labels.append(md.name)
            if not obj_paths:
                raise FileNotFoundError(
                    f"No model_normalized.obj files under {cat_dir}/*/models/.\n"
                    "Expected layout: {root}/{synset_id}/{model_id}/models/model_normalized.obj"
                )
            self._entries = obj_paths
            self._labels = labels
        else:
            raise ValueError(f"Unknown source {source!r}; use 'fallback' or 'shapenet'.")

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> o3d.geometry.TriangleMesh:
        if index < 0 or index >= len(self._entries):
            raise IndexError(f"Mesh index {index} out of range for dataset of size {len(self)}.")
        path = self._entries[index]
        try:
            mesh = load_mesh_file(path)
            return normalize_mesh_unit_sphere(mesh)
        except Exception as e:
            print(f"ERROR: failed to load mesh at index {index} ({path}): {e}", file=sys.stderr)
            raise

    def get_name(self, index: int) -> str:
        if index < 0 or index >= len(self._labels):
            raise IndexError(f"Mesh index {index} out of range for dataset of size {len(self)}.")
        return self._labels[index]
