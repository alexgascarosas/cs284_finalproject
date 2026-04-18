"""End-to-end smoke test for the recon backend.

Renders N synthetic RGB-D views of the Stanford bunny, fuses them into a TSDF,
extracts a mesh, and reports Chamfer distance vs. the original.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from recon import Reconstructor
from render import look_at, make_intrinsic, render_view


def fibonacci_sphere(n: int, radius: float = 1.0) -> np.ndarray:
    """Quasi-uniform points on a sphere; offset so we never hit the exact poles."""
    pts = np.zeros((n, 3))
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i + 0.5) / n * 2.0
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        pts[i] = (r * np.cos(theta), y, r * np.sin(theta))
    return pts * radius


def normalize_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    aabb = mesh.get_axis_aligned_bounding_box()
    mesh.translate(-aabb.get_center())
    mesh.scale(1.0 / max(aabb.get_extent()), center=(0.0, 0.0, 0.0))
    return mesh


def chamfer(a_pcd: o3d.geometry.PointCloud, b_pcd: o3d.geometry.PointCloud) -> float:
    d_ab = np.asarray(a_pcd.compute_point_cloud_distance(b_pcd))
    d_ba = np.asarray(b_pcd.compute_point_cloud_distance(a_pcd))
    return 0.5 * (float(d_ab.mean()) + float(d_ba.mean()))


def load_demo_mesh() -> o3d.geometry.TriangleMesh:
    """Stanford bunny if reachable; otherwise a procedural torus so the test runs offline."""
    try:
        bunny = o3d.data.BunnyMesh(data_root="data")
        mesh = o3d.io.read_triangle_mesh(bunny.path)
        if len(mesh.vertices) > 0:
            print("  using Stanford bunny")
            return mesh
    except Exception as exc:  # noqa: BLE001
        print(f"  bunny unavailable ({type(exc).__name__}); falling back to torus")
    print("  using procedural torus")
    return o3d.geometry.TriangleMesh.create_torus(
        torus_radius=1.0,
        tube_radius=0.4,
        radial_resolution=60,
        tubular_resolution=30,
    )


def main() -> None:
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    print("loading demo mesh...")
    mesh = load_demo_mesh()
    mesh.compute_vertex_normals()
    normalize_mesh(mesh)

    width, height = 320, 240
    intrinsic = make_intrinsic(width, height, fov_deg=55.0)
    n_views = 12
    radius = 1.8

    recon = Reconstructor(voxel_size=0.008, sdf_trunc=0.03, depth_max=4.0)

    tiles = []
    extrinsics: list[np.ndarray] = []
    for i, eye in enumerate(fibonacci_sphere(n_views, radius=radius)):
        extrinsic = look_at(eye)
        extrinsics.append(extrinsic)
        rgb, depth = render_view(mesh, intrinsic, extrinsic)
        recon.integrate(rgb, depth, intrinsic, extrinsic)

        d_max = max(float(depth.max()), 1e-6)
        depth_vis = (np.clip(depth / d_max, 0.0, 1.0) * 255.0).astype(np.uint8)
        tiles.append(np.concatenate([rgb, np.stack([depth_vis] * 3, axis=-1)], axis=1))

        valid = depth > 0
        d_min = float(depth[valid].min()) if valid.any() else 0.0
        print(f"  view {i + 1:>2}/{n_views} integrated  (depth {d_min:.2f}..{d_max:.2f})")

    cols = 4
    rows = (n_views + cols - 1) // cols
    tile_h, tile_w = height, width * 2
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
    o3d.io.write_image(str(out_dir / "views.png"), o3d.geometry.Image(canvas))

    out_mesh = recon.extract_mesh()
    o3d.io.write_triangle_mesh(str(out_dir / "recon.ply"), out_mesh)
    o3d.io.write_triangle_mesh(str(out_dir / "gt_mesh.ply"), mesh)

    K = np.asarray(intrinsic.intrinsic_matrix)
    np.savez(
        out_dir / "cameras.npz",
        extrinsics=np.stack(extrinsics, axis=0),
        width=intrinsic.width,
        height=intrinsic.height,
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
    )

    if len(out_mesh.vertices) == 0:
        raise RuntimeError("reconstructed mesh is empty; check rendering / TSDF parameters")

    gt_pcd = mesh.sample_points_uniformly(number_of_points=20000)
    recon_pcd = out_mesh.sample_points_uniformly(number_of_points=20000)
    cd = chamfer(gt_pcd, recon_pcd)

    print(f"\nrecon mesh: {len(out_mesh.vertices)} verts, {len(out_mesh.triangles)} tris")
    print(f"chamfer (mean of bidirectional means): {cd:.5f}  (mesh extent ~= 1.0)")
    print(f"wrote {out_dir}/{{recon.ply, gt_mesh.ply, cameras.npz, views.png}}")
    print("run `python visualize.py` to inspect interactively")


if __name__ == "__main__":
    main()
