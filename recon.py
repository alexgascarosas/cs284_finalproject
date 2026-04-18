"""Incremental TSDF reconstruction wrapper around Open3D."""
from __future__ import annotations

import numpy as np
import open3d as o3d


class Reconstructor:
    def __init__(
        self,
        voxel_size: float = 0.01,
        sdf_trunc: float = 0.04,
        depth_scale: float = 1.0,
        depth_max: float = 4.0,
    ):
        self.depth_scale = depth_scale
        self.depth_max = depth_max
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def integrate(self, rgb: np.ndarray, depth: np.ndarray, intrinsic, extrinsic: np.ndarray) -> None:
        """Fuse one RGB-D frame.

        rgb:       HxWx3 uint8.
        depth:     HxW float32, perpendicular z-depth in metric units.
        intrinsic: o3d.camera.PinholeCameraIntrinsic.
        extrinsic: 4x4 world-to-camera matrix.
        """
        color_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=self.depth_scale,
            depth_trunc=self.depth_max,
            convert_rgb_to_intensity=False,
        )
        self.volume.integrate(rgbd, intrinsic, extrinsic)

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    def extract_pointcloud(self) -> o3d.geometry.PointCloud:
        return self.volume.extract_point_cloud()
