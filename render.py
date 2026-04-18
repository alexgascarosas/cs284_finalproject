"""Offscreen RGB-D rendering of a known mesh from known camera poses, via Open3D raycasting.

Uses RaycastingScene (CPU, no GL context) so it runs headless on macOS / Linux / Windows
without surprises.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d


def make_intrinsic(width: int, height: int, fov_deg: float = 60.0) -> o3d.camera.PinholeCameraIntrinsic:
    f = 0.5 * height / np.tan(0.5 * np.deg2rad(fov_deg))
    cx, cy = width * 0.5 - 0.5, height * 0.5 - 0.5
    return o3d.camera.PinholeCameraIntrinsic(width, height, f, f, cx, cy)


def look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """4x4 world-to-camera extrinsic for an OpenCV-style camera (x right, y down, z forward)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    rot = np.stack([right, -true_up, forward], axis=0)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rot
    extrinsic[:3, 3] = -rot @ eye
    return extrinsic


def render_view(
    mesh: o3d.geometry.TriangleMesh,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    extrinsic: np.ndarray,
):
    """Render synthetic RGB and metric z-depth for `mesh` from a given camera pose.

    Returns (rgb HxWx3 uint8, depth HxW float32 perpendicular to image plane).
    """
    width, height = intrinsic.width, intrinsic.height
    K = np.asarray(intrinsic.intrinsic_matrix)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    rays = scene.create_rays_pinhole(
        intrinsic_matrix=K,
        extrinsic_matrix=extrinsic,
        width_px=width,
        height_px=height,
    )
    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()

    # t_hit is along-ray distance; convert to perpendicular z-depth in camera frame.
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x = (u - cx) / fx
    y = (v - cy) / fy
    inv_norm = 1.0 / np.sqrt(x * x + y * y + 1.0)
    depth = (t_hit * inv_norm).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0

    normals = ans["primitive_normals"].numpy()
    light = np.array([0.3, 0.7, -0.3])
    light /= np.linalg.norm(light)
    shading = np.clip(normals @ light, 0.15, 1.0)
    gray = (shading * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[depth <= 0] = 0
    return rgb, depth
