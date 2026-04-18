"""
Interactive Open3D preview for meshes from MeshDataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import open3d as o3d

from dataset import MeshDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview a mesh from MeshDataset in Open3D.")
    parser.add_argument(
        "--source",
        choices=("fallback", "shapenet"),
        required=True,
        help="Dataset source: fallback (data/meshes/*.ply) or shapenet.",
    )
    parser.add_argument("--index", type=int, default=0, help="Dataset index (default: 0).")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="ShapeNet root (ShapeNetCore.v2). Default: data/shapenet/ next to this script.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="ShapeNet synset ID folder name (required for --source shapenet).",
    )
    args = parser.parse_args()

    try:
        if args.source == "fallback":
            ds = MeshDataset(source="fallback")
        else:
            if not args.category:
                parser.error("--category is required when --source shapenet")
            ds = MeshDataset(
                source="shapenet",
                root=args.root,
                category=args.category,
            )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    idx = args.index
    try:
        name = ds.get_name(idx)
        mesh = ds[idx]
    except IndexError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: could not load mesh: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Showing index {idx}: {name} ({len(mesh.vertices)} vertices, "
        f"{len(mesh.triangles)} triangles)"
    )
    o3d.visualization.draw_geometries([mesh], window_name=f"preview: {name}")


if __name__ == "__main__":
    main()
