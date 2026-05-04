"""Synthetic next-best-view experiments for active 3D reconstruction.

Pipeline overview:
1. Load and normalize a ground-truth mesh.
2. Place candidate cameras on a sphere around the object.
3. Render RGB-D for every candidate and precompute which GT samples are visible.
4. Use a selection strategy to choose a subset of views over time.
5. Fuse the chosen views into a TSDF reconstruction.
6. Save checkpoint meshes, final mesh, and per-step metrics for comparison.

The key split in this file is:
- coverage metrics are computed from visible GT surface samples,
- geometry quality is computed from the TSDF-fused reconstruction mesh.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d

from recon import Reconstructor
from render import (
    inside_image_mask,
    look_at,
    make_intrinsic,
    project_camera_points,
    render_view,
    world_to_camera,
)


def fibonacci_sphere(n: int, radius: float = 1.0) -> np.ndarray:
    """Generate approximately uniform camera centers on a sphere."""
    pts = np.zeros((n, 3), dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i + 0.5) / n * 2.0
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        pts[i] = (r * np.cos(theta), y, r * np.sin(theta))
    return pts * radius


def normalize_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Center and scale a mesh to roughly unit extent for stable tuning."""
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
    """Symmetric Chamfer proxy used throughout the experiments."""
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
    """Load one of the fallback meshes used for repeatable local experiments."""
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
    """Save side-by-side RGB/depth tiles for the selected camera sequence."""
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
    """Build one montage tile with RGB on the left and normalized depth on the right."""
    d_max = max(float(depth.max()), 1e-6)
    depth_vis = (np.clip(depth / d_max, 0.0, 1.0) * 255.0).astype(np.uint8)
    depth_vis_rgb = np.stack([depth_vis] * 3, axis=-1)
    return np.concatenate([rgb, depth_vis_rgb], axis=1)


def visible_sample_mask(
    points_world: np.ndarray,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    extrinsic: np.ndarray,
    depth_map: np.ndarray,
    z_tol: float = 0.02,
    patch_radius: int = 1,
) -> np.ndarray:
    """Estimate which GT samples are visible in a rendered depth map.

    A sample is considered visible when:
    - it projects in front of the camera and inside the image,
    - the local depth patch contains at least one valid rendered depth value,
    - the sample lies no farther than `depth_tol` behind the nearest valid depth
      in that patch.

    This uses the same projection helpers as `render_view` so rendering and
    visibility scoring share one camera convention.
    """
    cam = world_to_camera(points_world, extrinsic)
    z = cam[:, 2]
    width, height = intrinsic.width, intrinsic.height

    mask = np.zeros(len(points_world), dtype=bool)
    u, v, valid = project_camera_points(cam, intrinsic)
    inside = valid & inside_image_mask(u, v, width, height)
    idx_inside = np.flatnonzero(inside)
    if len(idx_inside) == 0:
        return mask

    nearest_depth = np.zeros(len(idx_inside), dtype=np.float32)
    for out_idx, point_idx in enumerate(idx_inside):
        u0 = max(0, u[point_idx] - patch_radius)
        u1 = min(width, u[point_idx] + patch_radius + 1)
        v0 = max(0, v[point_idx] - patch_radius)
        v1 = min(height, v[point_idx] + patch_radius + 1)
        patch = depth_map[v0:v1, u0:u1]
        valid_patch = patch[patch > 0.0]
        if valid_patch.size > 0:
            nearest_depth[out_idx] = float(valid_patch.min())

    good_depth = nearest_depth > 0.0
    mask[idx_inside] = good_depth & (z[idx_inside] <= nearest_depth + z_tol)
    return mask


@dataclass
class CandidateView:
    """Cached render + visibility information for one candidate camera pose."""
    idx: int
    eye: np.ndarray
    extrinsic: np.ndarray
    rgb: np.ndarray
    depth: np.ndarray
    visible_mask: np.ndarray


class NBVSelector:
    """Base interface for policies that choose the next unused candidate view."""
    def select(self, candidates: list[CandidateView], seen_counts: np.ndarray, unused: np.ndarray) -> int:
        raise NotImplementedError


class RandomSelector(NBVSelector):
    """Unbiased baseline: choose any remaining candidate uniformly at random."""
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def select(self, candidates: list[CandidateView], seen_counts: np.ndarray, unused: np.ndarray) -> int:
        del candidates, seen_counts
        available = np.flatnonzero(unused)
        return int(self.rng.choice(available))


class CoverageSelector(NBVSelector):
    """Prefer views that expose the most currently unseen GT samples."""
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
    """Prefer views that revisit already-seen surface with low observation count."""
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

class HybridSelector(NBVSelector):
    """Weighted exploration/refinement policy.

    Coverage rewards newly visible GT surface samples, so it drives fast early
    growth. Uncertainty rewards already-seen samples with low observation count,
    so it adds redundant views that can improve fused geometry. The two score
    families are normalized per step before weighting so raw sample-count scale
    does not make either term dominate.
    """
    def __init__(self, total_steps: int, alpha: float = 1.0, beta: float = 0.5):
        self.total_steps = total_steps
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        valid = np.isfinite(scores)
        out = np.zeros_like(scores, dtype=np.float64)
        if not np.any(valid):
            return out
        vals = scores[valid]
        lo, hi = float(vals.min()), float(vals.max())
        if hi > lo:
            out[valid] = (vals - lo) / (hi - lo)
        return out

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
            coverage_scores[i] = float(np.sum(vis & (seen_counts == 0)))

            refine_mask = vis & (seen_counts > 0)
            if np.any(refine_mask):
                uncertainty_scores[i] = float(np.sum(1.0 / (1.0 + seen_counts[refine_mask])))
            else:
                uncertainty_scores[i] = 0.0

        coverage_norm = self._normalize(coverage_scores)
        uncertainty_norm = self._normalize(uncertainty_scores)

        progress = step / max(1, self.total_steps - 1)
        coverage_weight = self.alpha * (1.0 - 0.6 * progress)
        uncertainty_weight = self.beta * (0.4 + 0.6 * progress)
        hybrid_score = coverage_weight * coverage_norm + uncertainty_weight * uncertainty_norm
        hybrid_score[~unused] = -np.inf

        return int(np.argmax(hybrid_score))

def build_selector(strategy: str, rng: np.random.Generator, alpha: float, beta: float, total_steps: int) -> NBVSelector:
    """Instantiate the requested view-selection policy."""
    if strategy == "random":
        return RandomSelector(rng)
    if strategy == "coverage":
        return CoverageSelector()
    if strategy == "uncertainty":
        return UncertaintySelector()
    if strategy == "hybrid":
        return HybridSelector(total_steps=total_steps, alpha=alpha, beta=beta)
    raise ValueError(f"unknown strategy: {strategy}")


def precompute_candidates(
    mesh: o3d.geometry.TriangleMesh,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    candidate_eyes: np.ndarray,
    gt_samples: np.ndarray,
    depth_tol: float,
    patch_radius: int,
) -> list[CandidateView]:
    """Render and cache every candidate view before the selection loop starts."""
    candidates: list[CandidateView] = []
    print(f"precomputing {len(candidate_eyes)} candidate views...", flush=True)
    for i, eye in enumerate(candidate_eyes):
        extrinsic = look_at(eye)
        rgb, depth = render_view(mesh, intrinsic, extrinsic)
        vis = visible_sample_mask(
            gt_samples,
            intrinsic,
            extrinsic,
            depth,
            z_tol=depth_tol,
            patch_radius=patch_radius,
        )
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
    """Convert visible GT samples into a downsampled point cloud for coverage metrics."""
    pcd = o3d.geometry.PointCloud()
    if not np.any(mask):
        return pcd
    pcd.points = o3d.utility.Vector3dVector(points[mask])
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


def evaluate_mask_chamfer(
    points_world: np.ndarray,
    mask: np.ndarray,
    gt_mesh: o3d.geometry.TriangleMesh,
    n_points: int = 20000,
) -> tuple[float, int]:
    """Evaluate GT-sample coverage as a point-cloud Chamfer proxy."""
    recon_pcd = pointcloud_from_mask(points_world, mask)
    if len(recon_pcd.points) == 0:
        return float("inf"), 0
    gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=n_points)
    cd = chamfer(gt_pcd, recon_pcd)
    return cd, len(recon_pcd.points)


def evaluate_mesh_chamfer(
    recon_mesh: o3d.geometry.TriangleMesh,
    gt_mesh: o3d.geometry.TriangleMesh,
    n_points: int = 20000,
) -> tuple[float, int, int]:
    """Evaluate the actual TSDF reconstruction mesh against the GT mesh."""
    if len(recon_mesh.vertices) == 0 or len(recon_mesh.triangles) == 0:
        return float("inf"), 0, 0
    gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=n_points)
    recon_pcd = recon_mesh.sample_points_uniformly(number_of_points=n_points)
    return chamfer(gt_pcd, recon_pcd), len(recon_mesh.vertices), len(recon_mesh.triangles)


def parse_save_steps(raw_steps: str, n_steps: int) -> list[int]:
    """Parse a comma-separated checkpoint list and always include the final step."""
    steps = set()
    for token in raw_steps.split(","):
        token = token.strip()
        if not token:
            continue
        step = int(token)
        if step > 0:
            steps.add(min(step, n_steps))
    steps.add(n_steps)
    return sorted(steps)


def build_run_root(out_root: Path, mesh_name: str, run_name: str | None) -> Path:
    """Create a unique run directory so artifacts do not overwrite each other."""
    if run_name:
        folder = run_name
    else:
        folder = f"{mesh_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = out_root / folder
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


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
    save_steps: list[int],
) -> None:
    """Run one strategy end-to-end and write its artifacts.

    Important outputs:
    - `recon_step_XX.ply`: checkpoint TSDF meshes
    - `recon.ply`: final TSDF mesh
    - `views.png`: selected RGB/depth montage
    - `metrics.csv`: coverage and geometry quality over time
    """
    selector = build_selector(strategy, rng=rng, alpha=alpha, beta=beta, total_steps=n_steps)
    out_dir.mkdir(parents=True, exist_ok=True)

    o3d.io.write_triangle_mesh(str(out_dir / "gt_mesh.ply"), mesh)

    seen_counts = np.zeros(len(gt_samples), dtype=np.int32)
    recon_mask = np.zeros(len(gt_samples), dtype=bool)
    unused = np.ones(len(candidates), dtype=bool)
    recon = Reconstructor(voxel_size=0.008, sdf_trunc=0.03, depth_max=4.0)

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

        # Coverage and geometry are tracked separately: visibility updates the
        # sample-based metrics, while TSDF fusion updates the actual mesh.
        recon.integrate(cand.rgb, cand.depth, intrinsic, cand.extrinsic)
        seen_counts[cand.visible_mask] += 1
        recon_mask |= cand.visible_mask

        selected_extrinsics.append(cand.extrinsic.copy())
        selected_tiles.append(render_tile(cand.rgb, cand.depth))

        coverage_cd, n_pts = evaluate_mask_chamfer(gt_samples, recon_mask, mesh)
        recon_mesh = recon.extract_mesh()
        mesh_cd, mesh_verts, mesh_tris = evaluate_mesh_chamfer(recon_mesh, mesh)
        seen_frac = float(np.mean(recon_mask))
        avg_obs = float(np.mean(seen_counts[recon_mask])) if np.any(recon_mask) else 0.0
        newly_seen = int(np.sum(cand.visible_mask & (seen_counts == 1)))

        metrics_rows.append(
            {
                "step": step + 1,
                "candidate_idx": idx,
                "seen_fraction": seen_frac,
                "avg_observations_seen_surface": avg_obs,
                "newly_seen_samples": newly_seen,
                "coverage_chamfer": coverage_cd,
                "recon_points": n_pts,
                "mesh_chamfer": mesh_cd,
                "mesh_vertices": mesh_verts,
                "mesh_triangles": mesh_tris,
            }
        )

        print(
            f"  step {step + 1:>2}/{n_steps} "
            f"view={idx:>2} "
            f"new={newly_seen:>5} "
            f"seen={seen_frac:.3f} "
            f"mesh_cd={mesh_cd:.5f} "
            f"pts={n_pts:>6}",
            flush=True,
        )

        if (step + 1) in save_steps and len(recon_mesh.vertices) > 0:
            o3d.io.write_triangle_mesh(str(out_dir / f"recon_step_{step + 1:02d}.ply"), recon_mesh)

    final_mesh = recon.extract_mesh()
    if len(final_mesh.vertices) == 0 or len(final_mesh.triangles) == 0:
        raise RuntimeError(f"{strategy}: reconstructed mesh is empty")

    o3d.io.write_triangle_mesh(str(out_dir / "recon.ply"), final_mesh)
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
                "coverage_chamfer",
                "recon_points",
                "mesh_chamfer",
                "mesh_vertices",
                "mesh_triangles",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    print(f"\nfinished strategy={strategy}", flush=True)
    print(f"  final mesh chamfer: {metrics_rows[-1]['mesh_chamfer']:.5f}", flush=True)
    print(f"  wrote {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    """Command-line interface for local experiments and report generation."""
    parser = argparse.ArgumentParser(description="Run milestone next-best-view experiments.")
    parser.add_argument("--strategy", type=str, default="all", choices=["all", "random", "coverage", "uncertainty", "hybrid"])
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--num-candidates", type=int, default=40)
    parser.add_argument("--radius", type=float, default=2.2)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fov", type=float, default=55.0)
    parser.add_argument("--gt-samples", type=int, default=15000)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--depth-tol", type=float, default=0.02)
    parser.add_argument("--visibility-patch-radius", type=int, default=1)
    parser.add_argument("--save-steps", type=str, default="1,3,5,10")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-root", type=str, default="output_nbv")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
                            "--mesh",
                            type=str,
                            default="torus",
                            choices=["torus", "bunny", "armadillo", "dragon"],
                            help="Which mesh to run NBV on.",
                        )
    return parser.parse_args()


def main() -> None:
    """Build the experiment state, then run one or more selection strategies."""
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
    candidates = precompute_candidates(
        mesh,
        intrinsic,
        candidate_eyes,
        gt_samples,
        depth_tol=args.depth_tol,
        patch_radius=args.visibility_patch_radius,
    )

    strategies = ["random", "coverage", "uncertainty", "hybrid"] if args.strategy == "all" else [args.strategy]

    out_root = Path(args.out_root)
    out_root.mkdir(exist_ok=True)
    run_root = build_run_root(out_root, args.mesh, args.run_name)
    save_steps = parse_save_steps(args.save_steps, min(args.steps, len(candidates)))

    for strategy in strategies:
        print(f"starting strategy {strategy}...", flush=True)
        run_experiment(
            strategy=strategy,
            mesh=mesh,
            intrinsic=intrinsic,
            candidates=candidates,
            gt_samples=gt_samples,
            n_steps=min(args.steps, len(candidates)),
            out_dir=run_root / strategy,
            rng=rng,
            alpha=args.alpha,
            beta=args.beta,
            save_steps=save_steps,
        )

    print("\ndone.", flush=True)
    print(f"results saved under: {run_root}", flush=True)


if __name__ == "__main__":
    main()
