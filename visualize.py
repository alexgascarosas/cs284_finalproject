"""Interactive viewer for the smoke-test outputs.

Shows the reconstructed mesh, the ground-truth mesh (offset to the right), and
the camera frustums used during integration. Mouse to orbit, scroll to zoom.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def load_cameras(npz_path: Path):
    data = np.load(npz_path)
    intr = o3d.camera.PinholeCameraIntrinsic(
        int(data["width"]),
        int(data["height"]),
        float(data["fx"]),
        float(data["fy"]),
        float(data["cx"]),
        float(data["cy"]),
    )
    return intr, np.asarray(data["extrinsics"])


def camera_frustum(intrinsic, extrinsic, scale: float = 0.15, color=(0.9, 0.2, 0.2)):
    fr = o3d.geometry.LineSet.create_camera_visualization(
        view_width_px=intrinsic.width,
        view_height_px=intrinsic.height,
        intrinsic=intrinsic.intrinsic_matrix,
        extrinsic=extrinsic,
        scale=scale,
    )
    fr.paint_uniform_color(color)
    return fr


def main() -> None:
    out = Path("output")
    if not out.exists():
        raise SystemExit("output/ not found - run `python smoke_test.py` first")

    geometries: list = []

    recon_path = out / "recon.ply"
    recon = o3d.io.read_triangle_mesh(str(recon_path))
    if len(recon.vertices) == 0:
        raise SystemExit(f"{recon_path} is empty - re-run smoke_test.py")
    recon.compute_vertex_normals()
    recon.paint_uniform_color([0.95, 0.55, 0.20])
    geometries.append(recon)

    gt_path = out / "gt_mesh.ply"
    if gt_path.exists():
        gt = o3d.io.read_triangle_mesh(str(gt_path))
        gt.compute_vertex_normals()
        gt.translate((1.5, 0.0, 0.0))
        gt.paint_uniform_color([0.55, 0.60, 0.70])
        geometries.append(gt)
    else:
        print(f"  note: {gt_path} not found - showing recon only")

    cam_path = out / "cameras.npz"
    if cam_path.exists():
        intr, extrinsics = load_cameras(cam_path)
        for ext in extrinsics:
            geometries.append(camera_frustum(intr, ext))
    else:
        print(f"  note: {cam_path} not found - no camera frustums shown")

    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))

    print(
        "controls:\n"
        "  left-drag  : rotate\n"
        "  right-drag : pan\n"
        "  scroll     : zoom\n"
        "  Q / Esc    : quit\n"
        "legend: orange = recon | gray = ground truth | red = camera frustums\n"
    )

    o3d.visualization.draw_geometries(
        geometries,
        window_name="ActiveView - recon (orange) | GT (gray) | cameras (red)",
        width=1280,
        height=800,
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()
