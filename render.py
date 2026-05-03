"""Offscreen RGB-D rendering of a mesh from known camera poses.

This module is the single source of truth for camera conventions used by the
project. Both rendering and visibility scoring rely on the same helpers so
pixel projection, world-to-camera transforms, and depth semantics stay aligned.

Convention summary:
- `extrinsic` is always a 4x4 world-to-camera transform.
- camera axes follow an OpenCV-style convention: x right, y down, z forward.
- depth values are camera-space z depth, not along-ray distance.
- invalid depth pixels are encoded as 0.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d


def camera_intrinsic_params(intrinsic: o3d.camera.PinholeCameraIntrinsic) -> tuple[float, float, float, float]:
    """Return `(fx, fy, cx, cy)` from an Open3D pinhole intrinsic."""
    K = np.asarray(intrinsic.intrinsic_matrix)
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def make_intrinsic(width: int, height: int, fov_deg: float = 60.0) -> o3d.camera.PinholeCameraIntrinsic:
    """Create a symmetric pinhole intrinsic from image size and vertical FOV."""
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


def world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """Transform Nx3 world-space points into camera coordinates."""
    points_world = np.asarray(points_world, dtype=np.float64)
    pts_h = np.concatenate([points_world, np.ones((len(points_world), 1), dtype=np.float64)], axis=1)
    cam_h = (extrinsic @ pts_h.T).T
    return cam_h[:, :3]


def pixel_round(coords: np.ndarray) -> np.ndarray:
    """Round projected floating-point pixels with the shared project-wide rule."""
    return np.rint(coords).astype(np.int32)


def project_camera_points(
    points_camera: np.ndarray,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-space points to integer pixels and return validity mask.

    Returns `(u, v, valid_z)` where invalid-z points keep `u=v=-1`.
    """
    fx, fy, cx, cy = camera_intrinsic_params(intrinsic)
    z = points_camera[:, 2]
    valid = z > 1e-6

    u = np.full(len(points_camera), -1, dtype=np.int32)
    v = np.full(len(points_camera), -1, dtype=np.int32)
    if np.any(valid):
        u[valid] = pixel_round(fx * (points_camera[valid, 0] / z[valid]) + cx)
        v[valid] = pixel_round(fy * (points_camera[valid, 1] / z[valid]) + cy)
    return u, v, valid


def inside_image_mask(u: np.ndarray, v: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return a mask for integer pixel coordinates inside the image bounds."""
    return (u >= 0) & (u < width) & (v >= 0) & (v < height)


def rays_to_camera_z_depth(
    t_hit: np.ndarray,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
) -> np.ndarray:
    """Convert ray hit distances from Open3D into camera-space z depth.

    Open3D reports `t_hit` as distance along the cast ray. TSDF fusion and
    visibility checks in this project operate on perpendicular camera-space z,
    so this helper performs the conversion and zeros invalid pixels.
    """
    width, height = intrinsic.width, intrinsic.height
    fx, fy, cx, cy = camera_intrinsic_params(intrinsic)
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x = (u - cx) / fx
    y = (v - cy) / fy
    inv_norm = 1.0 / np.sqrt(x * x + y * y + 1.0)

    depth = np.zeros_like(t_hit, dtype=np.float32)
    valid = np.isfinite(t_hit) & (t_hit > 0.0)
    depth[valid] = (t_hit[valid] * inv_norm[valid]).astype(np.float32)
    return depth


def render_view(
    mesh: o3d.geometry.TriangleMesh,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    extrinsic: np.ndarray,
):
    """Render synthetic RGB and metric z-depth for `mesh` from a given camera pose.

    Returns:
    - `rgb`: shaded grayscale image stored as `HxWx3 uint8`
    - `depth`: `HxW float32` camera-space z depth with invalid pixels set to `0`
    """
    width, height = intrinsic.width, intrinsic.height
    K = np.asarray(intrinsic.intrinsic_matrix)

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

    # Convert along-ray hit distance to perpendicular camera-space z depth.
    depth = rays_to_camera_z_depth(t_hit, intrinsic)

    normals = ans["primitive_normals"].numpy()
    light = np.array([0.3, 0.7, -0.3])
    light /= np.linalg.norm(light)
    shading = np.clip(normals @ light, 0.15, 1.0)
    gray = (shading * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[depth <= 0] = 0
    return rgb, depth
