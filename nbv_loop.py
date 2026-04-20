"""Next-best-view loop (milestone-stable fallback).

This version avoids unstable Open3D TSDF / depth integration on Windows by
reconstructing a fused point cloud from the union of visible GT surface samples.

That is enough for milestone experiments:
- random / coverage / uncertainty / hybrid view selection
- progressive reconstruction
- Chamfer-vs-step metrics
- saved point clouds and camera trajectories
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from render import look_at, make_intrinsic, render_view


def fibonacci_sphere(n: int, radius: float = 1.0) -> np.ndarray:
    pts = np.zeros((n, 3), dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i + 0.5) / n * 2.0
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        pts[i] = (r * np.cos(theta), y, r * np.sin(theta))
    return pts * radius


def normalize_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    verts = np.asarray(mesh.vertices).copy()
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    center = 0.5 * (vmin + vmax)
    extent = vmax - vmin
    scale = 1.0 / float(np.max(extent))
    verts = (verts - center) * scale
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    return mesh


def chamfer(a_pcd: o3d.geometry.PointCloud, b_pcd: o3d.geometry.PointCloud) -> float:
    d_ab = np.asarray(a_pcd.compute_point_cloud_distance(b_pcd))
    d_ba = np.asarray(b_pcd.compute_point_cloud_distance(a_pcd))
    return 0.5 * (float(d_ab.mean()) + float(d_ba.mean()))


# def load_demo_mesh() -> o3d.geometry.TriangleMesh:

#     # try:
#     #     bunny = o3d.data.BunnyMesh(data_root="data")
#     #     mesh = o3d.io.read_triangle_mesh(bunny.path)

#     #     if len(mesh.vertices) > 0:
#     #         print("  using Stanford bunny")
#     #         return mesh
#     # except Exception as e:
#     #     print("  bunny failed, fallback to torus:", e)

#     # # fallback
#     # return o3d.geometry.TriangleMesh.create_torus(
#     #     torus_radius=1.0,
#     #     tube_radius=0.4,
#     #     radial_resolution=60,
#     #     tubular_resolution=30,
#     # )
#     mesh_path = Path("data/meshes") / f"{mesh_name}.ply"
#     if mesh_path.exists():
#         print(f"  using fallback mesh: {mesh_name}")
#         mesh = o3d.io.read_triangle_mesh(str(mesh_path))
#         if len(mesh.vertices) > 0:
#             return mesh

#     raise FileNotFoundError(f"Could not load mesh '{mesh_name}' from {mesh_path}")

def load_demo_mesh(mesh_name: str = "torus"):
    if mesh_name == "torus":
        print("  using procedural torus")
        return o3d.geometry.TriangleMesh.create_torus(
            torus_radius=1.0,
            tube_radius=0.4,
            radial_resolution=60,
            tubular_resolution=30,
        )

    mesh_path = Path("data/meshes") / f"{mesh_name}.ply"

    if mesh_path.exists():
        print(f"  using fallback mesh: {mesh_name}")
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))

        if len(mesh.vertices) > 0:
            return mesh

    raise FileNotFoundError(f"Could not load mesh '{mesh_name}' from {mesh_path}")


def save_view_montage(tiles: list[np.ndarray], width: int, height: int, out_path: Path) -> None:
    if not tiles:
        return
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    tile_h, tile_w = height, width * 2
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
    o3d.io.write_image(str(out_path), o3d.geometry.Image(canvas))


def render_tile(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    d_max = max(float(depth.max()), 1e-6)
    depth_vis = (np.clip(depth / d_max, 0.0, 1.0) * 255.0).astype(np.uint8)
    depth_vis_rgb = np.stack([depth_vis] * 3, axis=-1)
    return np.concatenate([rgb, depth_vis_rgb], axis=1)


def world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([points_world, np.ones((len(points_world), 1), dtype=np.float64)], axis=1)
    cam_h = (extrinsic @ pts_h.T).T
    return cam_h[:, :3]


def visible_sample_mask(
    points_world: np.ndarray,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    extrinsic: np.ndarray,
    depth_map: np.ndarray,
    z_tol: float = 0.01,
) -> np.ndarray:
    cam = world_to_camera(points_world, extrinsic)
    z = cam[:, 2]

    K = np.asarray(intrinsic.intrinsic_matrix)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    width, height = intrinsic.width, intrinsic.height

    valid = z > 1e-6
    u = np.full(len(points_world), -1, dtype=np.int32)
    v = np.full(len(points_world), -1, dtype=np.int32)

    u_proj = np.round(fx * (cam[valid, 0] / z[valid]) + cx).astype(np.int32)
    v_proj = np.round(fy * (cam[valid, 1] / z[valid]) + cy).astype(np.int32)

    inside = (u_proj >= 0) & (u_proj < width) & (v_proj >= 0) & (v_proj < height)

    idx_valid = np.where(valid)[0]
    idx_inside = idx_valid[inside]
    u[idx_inside] = u_proj[inside]
    v[idx_inside] = v_proj[inside]

    mask = np.zeros(len(points_world), dtype=bool)
    if len(idx_inside) == 0:
        return mask

    sampled_depth = depth_map[v[idx_inside], u[idx_inside]]
    good_depth = sampled_depth > 0.0
    z_match = np.abs(z[idx_inside] - sampled_depth) <= z_tol
    mask[idx_inside] = good_depth & z_match
    return mask


@dataclass
class CandidateView:
    idx: int
    eye: np.ndarray
    extrinsic: np.ndarray
    rgb: np.ndarray
    depth: np.ndarray
    visible_mask: np.ndarray


class NBVSelector:
    def select(self, candidates: list[CandidateView], seen_counts: np.ndarray, unused: np.ndarray) -> int:
        raise NotImplementedError


class RandomSelector(NBVSelector):
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def select(self, candidates: list[CandidateView], seen_counts: np.ndarray, unused: np.ndarray) -> int:
        del candidates, seen_counts
        available = np.flatnonzero(unused)
        return int(self.rng.choice(available))


class CoverageSelector(NBVSelector):
    def select(self, candidates: list[CandidateView], seen_counts: np.ndarray, unused: np.ndarray) -> int:
        best_idx = -1
        best_score = -np.inf
        for i, cand in enumerate(candidates):
            if not unused[i]:
                continue
            gain = np.sum(cand.visible_mask & (seen_counts == 0))
            if gain > best_score:
                best_score = float(gain)
                best_idx = i
        return best_idx


class UncertaintySelector(NBVSelector):
    def select(self, candidates: list[CandidateView], seen_counts: np.ndarray, unused: np.ndarray) -> int:
        best_idx = -1
        best_score = -np.inf
        for i, cand in enumerate(candidates):
            if not unused[i]:
                continue
            vis = cand.visible_mask
            if not np.any(vis):
                score = -np.inf
            else:
                #score = float(np.sum(1.0 / (1.0 + seen_counts[vis])))
                refine_mask = vis & (seen_counts > 0)
                if np.any(refine_mask):
                    score = float(np.sum(1.0 / (1.0 + seen_counts[refine_mask])))
                else:
                    score = 0.0
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

"""
class HybridSelector(NBVSelector):
    def __init__(self, alpha: float = 0.3, beta: float = 3.0):
        self.alpha = alpha
        self.beta = beta

    def select(self, candidates: list[CandidateView], seen_counts: np.ndarray, unused: np.ndarray) -> int:
        best_idx = -1
        best_score = -np.inf
        for i, cand in enumerate(candidates):
            if not unused[i]:
                continue
            vis = cand.visible_mask
            coverage_gain = float(np.sum(vis & (seen_counts == 0)))
            uncertainty_gain = np.sum((seen_counts[vis] > 0) * (1.0 / (seen_counts[vis] + 1)))
            score = self.alpha * coverage_gain + self.beta * uncertainty_gain
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx
"""
"""
class HybridSelector(NBVSelector):
    def __init__(self, total_steps: int, explore_bias: float = 0.7):
        self.total_steps = total_steps
        self.explore_bias = explore_bias

    def select(
        self,
        candidates: list[CandidateView],
        seen_counts: np.ndarray,
        unused: np.ndarray,
        step: int = 0,
    ) -> int:
        coverage_scores = np.full(len(candidates), -np.inf, dtype=np.float64)
        uncertainty_scores = np.full(len(candidates), -np.inf, dtype=np.float64)

        for i, cand in enumerate(candidates):
            if not unused[i]:
                continue

            vis = cand.visible_mask

            # coverage: unseen points only
            coverage_scores[i] = float(np.sum(vis & (seen_counts == 0)))

            # uncertainty: refine already-seen but low-confidence points
            refine_mask = vis & (seen_counts > 0)
            if np.any(refine_mask):
                    uncertainty_scores[i] = float(
                        np.sum(1.0 / (1.0 + seen_counts[refine_mask]))
                    )
            else:
                    uncertainty_scores[i] = 0.0

        # normalize each score family so one does not dominate by scale
        def normalize(x: np.ndarray) -> np.ndarray:
            valid = np.isfinite(x)
            if not np.any(valid):
                return np.zeros_like(x)
            xv = x[valid]
            xmin, xmax = xv.min(), xv.max()
            out = np.zeros_like(x)
            if xmax > xmin:
                out[valid] = (xv - xmin) / (xmax - xmin)
            return out

        coverage_norm = normalize(coverage_scores)
        uncertainty_norm = normalize(uncertainty_scores)

        # anneal from exploration -> refinement over time
        t = step / max(1, self.total_steps - 1)
        w_cov = 1.5 * (1.0 - t)
        w_unc = self.explore_bias + 1.5 * t

        hybrid_score = w_cov * coverage_norm + w_unc * uncertainty_norm
        hybrid_score[~unused] = -np.inf

        return int(np.argmax(hybrid_score))
"""
class HybridSelector(NBVSelector):
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.coverage_selector = CoverageSelector()
        self.uncertainty_selector = UncertaintySelector()

    def select(
        self,
        candidates: list[CandidateView],
        seen_counts: np.ndarray,
        unused: np.ndarray,
        step: int = 0,
    ) -> int:
        switch_step = self.total_steps // 2

        # First half: explore (coverage)
        if step < switch_step:
            return self.coverage_selector.select(candidates, seen_counts, unused)

        # Second half: refine (uncertainty)
        return self.uncertainty_selector.select(candidates, seen_counts, unused)

def build_selector(strategy: str, rng: np.random.Generator, alpha: float, beta: float, total_steps: int) -> NBVSelector:
    if strategy == "random":
        return RandomSelector(rng)
    if strategy == "coverage":
        return CoverageSelector()
    if strategy == "uncertainty":
        return UncertaintySelector()
    if strategy == "hybrid":
        return HybridSelector(total_steps=total_steps)
    raise ValueError(f"unknown strategy: {strategy}")


def precompute_candidates(
    mesh: o3d.geometry.TriangleMesh,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    candidate_eyes: np.ndarray,
    gt_samples: np.ndarray,
) -> list[CandidateView]:
    candidates: list[CandidateView] = []
    print(f"precomputing {len(candidate_eyes)} candidate views...", flush=True)
    for i, eye in enumerate(candidate_eyes):
        extrinsic = look_at(eye)
        rgb, depth = render_view(mesh, intrinsic, extrinsic)
        vis = visible_sample_mask(gt_samples, intrinsic, extrinsic, depth)
        candidates.append(
            CandidateView(
                idx=i,
                eye=eye,
                extrinsic=extrinsic,
                rgb=rgb,
                depth=depth,
                visible_mask=vis,
            )
        )
        print(f"  candidate {i + 1:>2}/{len(candidate_eyes)} visible_samples={int(vis.sum()):>5}", flush=True)
    return candidates


def pointcloud_from_mask(points: np.ndarray, mask: np.ndarray, voxel_size: float = 0.008) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    if not np.any(mask):
        return pcd
    pcd.points = o3d.utility.Vector3dVector(points[mask])
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


def evaluate_recon_mask(
    gt_samples: np.ndarray,
    recon_mask: np.ndarray,
    gt_mesh: o3d.geometry.TriangleMesh,
    n_points: int = 20000,
) -> tuple[float, int]:
    recon_pcd = pointcloud_from_mask(gt_samples, recon_mask)
    if len(recon_pcd.points) == 0:
        return float("inf"), 0
    gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=n_points)
    cd = chamfer(gt_pcd, recon_pcd)
    return cd, len(recon_pcd.points)


def run_experiment(
    strategy: str,
    mesh: o3d.geometry.TriangleMesh,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    candidates: list[CandidateView],
    gt_samples: np.ndarray,
    n_steps: int,
    out_dir: Path,
    rng: np.random.Generator,
    alpha: float,
    beta: float,
) -> None:
    selector = build_selector(strategy, rng=rng, alpha=alpha, beta=beta, total_steps=n_steps)
    out_dir.mkdir(parents=True, exist_ok=True)

    o3d.io.write_triangle_mesh(str(out_dir / "gt_mesh.ply"), mesh)

    seen_counts = np.zeros(len(gt_samples), dtype=np.int32)
    recon_mask = np.zeros(len(gt_samples), dtype=bool)
    unused = np.ones(len(candidates), dtype=bool)

    selected_extrinsics: list[np.ndarray] = []
    selected_tiles: list[np.ndarray] = []
    metrics_rows: list[dict[str, float | int]] = []

    print(f"\nrunning strategy={strategy} for {n_steps} steps...", flush=True)
    for step in range(n_steps):
        if strategy == "hybrid":
            idx = selector.select(candidates, seen_counts, unused, step=step)
        else:
            idx = selector.select(candidates, seen_counts, unused)
        cand = candidates[idx]
        unused[idx] = False

        seen_counts[cand.visible_mask] += 1
        recon_mask |= cand.visible_mask

        selected_extrinsics.append(cand.extrinsic.copy())
        selected_tiles.append(render_tile(cand.rgb, cand.depth))

        cd, n_pts = evaluate_recon_mask(gt_samples, recon_mask, mesh)
        seen_frac = float(np.mean(recon_mask))
        avg_obs = float(np.mean(seen_counts[recon_mask])) if np.any(recon_mask) else 0.0
        newly_seen = int(np.sum(cand.visible_mask & (~(recon_mask ^ cand.visible_mask))))

        metrics_rows.append(
            {
                "step": step + 1,
                "candidate_idx": idx,
                "seen_fraction": seen_frac,
                "avg_observations_seen_surface": avg_obs,
                "newly_seen_samples": int(np.sum(cand.visible_mask & (seen_counts == 1))),
                "chamfer": cd,
                "recon_points": n_pts,
            }
        )

        print(
            f"  step {step + 1:>2}/{n_steps} "
            f"view={idx:>2} "
            f"new={int(np.sum(cand.visible_mask & (seen_counts == 1))):>5} "
            f"seen={seen_frac:.3f} "
            f"cd={cd:.5f} "
            f"pts={n_pts:>6}",
            flush=True,
        )

        if (step + 1) in {1, 3, 5, 10, n_steps}:
            pcd_step = pointcloud_from_mask(gt_samples, recon_mask)
            o3d.io.write_point_cloud(str(out_dir / f"recon_step_{step + 1:02d}.ply"), pcd_step)

    final_pcd = pointcloud_from_mask(gt_samples, recon_mask)
    if len(final_pcd.points) == 0:
        raise RuntimeError(f"{strategy}: reconstructed point cloud is empty")

    o3d.io.write_point_cloud(str(out_dir / "recon.ply"), final_pcd)
    save_view_montage(selected_tiles, intrinsic.width, intrinsic.height, out_dir / "views.png")

    K = np.asarray(intrinsic.intrinsic_matrix)
    np.savez(
        out_dir / "cameras.npz",
        extrinsics=np.stack(selected_extrinsics, axis=0),
        width=intrinsic.width,
        height=intrinsic.height,
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
    )

    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "candidate_idx",
                "seen_fraction",
                "avg_observations_seen_surface",
                "newly_seen_samples",
                "chamfer",
                "recon_points",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    print(f"\nfinished strategy={strategy}", flush=True)
    print(f"  final chamfer: {metrics_rows[-1]['chamfer']:.5f}", flush=True)
    print(f"  wrote {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run milestone next-best-view experiments.")
    parser.add_argument("--strategy", type=str, default="all", choices=["all", "random", "coverage", "uncertainty", "hybrid"])
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--num-candidates", type=int, default=40)
    parser.add_argument("--radius", type=float, default=1.8)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fov", type=float, default=55.0)
    parser.add_argument("--gt-samples", type=int, default=15000)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-root", type=str, default="output_nbv")
    parser.add_argument(
                            "--mesh",
                            type=str,
                            default="torus",
                            choices=["torus", "bunny", "armadillo", "dragon"],
                            help="Which mesh to run NBV on.",
                        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("loading demo mesh...", flush=True)
    #mesh = load_demo_mesh()
    mesh = load_demo_mesh(args.mesh)

    print("normalizing mesh...", flush=True)
    normalize_mesh(mesh)

    print("making intrinsics...", flush=True)
    intrinsic = make_intrinsic(args.width, args.height, fov_deg=args.fov)

    print(f"sampling gt surface ({args.gt_samples})...", flush=True)
    gt_pcd = mesh.sample_points_uniformly(number_of_points=args.gt_samples)
    gt_samples = np.asarray(gt_pcd.points)

    print(f"building candidate eyes ({args.num_candidates})...", flush=True)
    candidate_eyes = fibonacci_sphere(args.num_candidates, radius=args.radius)

    print("precomputing candidates...", flush=True)
    candidates = precompute_candidates(mesh, intrinsic, candidate_eyes, gt_samples)

    strategies = ["random", "coverage", "uncertainty", "hybrid"] if args.strategy == "all" else [args.strategy]

    out_root = Path(args.out_root)
    out_root.mkdir(exist_ok=True)

    for strategy in strategies:
        print(f"starting strategy {strategy}...", flush=True)
        run_experiment(
            strategy=strategy,
            mesh=mesh,
            intrinsic=intrinsic,
            candidates=candidates,
            gt_samples=gt_samples,
            n_steps=min(args.steps, len(candidates)),
            out_dir=out_root / strategy,
            rng=rng,
            alpha=args.alpha,
            beta=args.beta,
        )

    print("\ndone.", flush=True)
    print(f"results saved under: {out_root}", flush=True)


if __name__ == "__main__":
    main()