"""
Download Stanford Bunny, Dragon, and Armadillo meshes from the
Stanford 3D Scanning Repository into data/meshes/ as .ply files.

Dependencies: requests, standard library (tarfile, gzip).
"""

from __future__ import annotations

import gzip
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent
MESH_DIR = DATA_DIR / "meshes"

# Official repository URLs (see https://graphics.stanford.edu/data/3Dscanrep/)
ARCHIVES: list[tuple[str, str, str]] = [
    (
        "bunny",
        "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz",
        "bunny/reconstruction/bun_zipper.ply",
    ),
    (
        "dragon",
        "http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz",
        "dragon/reconstruction/dragon_vrip.ply",
    ),
]


def _download_with_progress(url: str, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / (dest.name + ".part")
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length") or 0)
            downloaded = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = 100.0 * downloaded / total
                        print(
                            f"  [{label}] {downloaded / (1 << 20):.1f} / {total / (1 << 20):.1f} MB ({pct:.1f}%)",
                            end="\r",
                        )
                    else:
                        print(f"  [{label}] {downloaded / (1 << 20):.1f} MB", end="\r")
            print(f"  [{label}] saved {downloaded / (1 << 20):.1f} MB          ")
        os.replace(tmp, dest)
    except requests.RequestException:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _find_ply_member(tar: tarfile.TarFile, preferred_substrings: tuple[str, ...]) -> tarfile.TarInfo:
    members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".ply")]
    for sub in preferred_substrings:
        for m in members:
            if sub in m.name.replace("\\", "/").lower():
                return m
    if members:
        raise FileNotFoundError(
            "Could not find expected .ply in archive. "
            f"Tried substrings {preferred_substrings!r}. "
            f"Found {len(members)} .ply files; first: {members[0].name!r}"
        )
    raise FileNotFoundError("Archive contains no .ply files.")


def _extract_ply_from_tar(tar_path: Path, member_path: str, out_ply: Path, fallback_substrings: tuple[str, ...]) -> None:
    with tarfile.open(tar_path, "r:gz") as tar:
        try:
            member = tar.getmember(member_path)
            if not member.isfile():
                raise KeyError("not a file")
        except KeyError:
            member = _find_ply_member(tar, fallback_substrings)
        r = tar.extractfile(member)
        if r is None:
            raise RuntimeError(f"Could not read archive member {member.name!r}.")
        out_ply.parent.mkdir(parents=True, exist_ok=True)
        with open(out_ply, "wb") as f:
            shutil.copyfileobj(r, f)


def _download_armadillo_ply(out_ply: Path) -> None:
    url = "http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz"
    gz_dest = MESH_DIR / "_Armadillo.ply.gz"
    _download_with_progress(url, gz_dest, "armadillo.gz")
    try:
        print("  [armadillo] decompressing .ply.gz -> .ply ...")
        with gzip.open(gz_dest, "rb") as gz, open(out_ply, "wb") as ply:
            shutil.copyfileobj(gz, ply)
    except (gzip.BadGzipFile, OSError) as e:
        print(f"ERROR: failed to decompress Armadillo: {e}", file=sys.stderr)
        raise
    finally:
        gz_dest.unlink(missing_ok=True)


def main() -> None:
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {MESH_DIR}")
    print("Downloading Stanford Bunny and Dragon (tar.gz), Armadillo (.ply.gz)...\n")

    with tempfile.TemporaryDirectory(prefix="stanford_mesh_") as td:
        tmp = Path(td)
        for key, url, inner in ARCHIVES:
            tar_path = tmp / f"{key}.tar.gz"
            print(f"-> {key}: {url}")
            _download_with_progress(url, tar_path, key)
            out = MESH_DIR / f"{key}.ply"
            print(f"  [{key}] extracting {inner} -> {out.name}")
            fallback = ("bun_zipper",) if key == "bunny" else ("dragon_vrip", "vrip")
            _extract_ply_from_tar(tar_path, inner, out, fallback_substrings=fallback)
            print(f"  [{key}] done: {out}\n")

    print("-> armadillo: http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz")
    arm_out = MESH_DIR / "armadillo.ply"
    _download_armadillo_ply(arm_out)
    print(f"  [armadillo] done: {arm_out}\n")

    print("All fallback meshes saved under data/meshes/:")
    for p in sorted(MESH_DIR.glob("*.ply")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    try:
        main()
    except (requests.RequestException, OSError, tarfile.TarError, EOFError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
