# ActiveView: Hybrid View Selection for Efficient 3D Reconstruction

Course project for **CS184/284A — Computer Graphics and Imaging** (Spring 2026) at UC Berkeley.

We study **active view selection** for multi-view 3D reconstruction: an agent chooses camera viewpoints to improve reconstruction quality with fewer observations than dense or random sampling. Our planned approach combines **coverage-based exploration** and **uncertainty-based refinement** in a hybrid Next Best View (NBV) scoring function, evaluated against random, coverage-only, and uncertainty-only baselines.

## Team

- Alexander Gasca Rosas  
- Winfred Wang  
- Dennis Liang  
- Yousif Yacoby  

## Repository contents

| Item | Description |
|------|-------------|
| [`index.html`](index.html) | Project proposal (browser; MathJax and Google Fonts from CDNs) |
| [`data/`](data/) | Mesh dataset helpers: download Stanford meshes, `MeshDataset`, preview ([`data/README.md`](data/README.md)) |
| [`requirements.txt`](requirements.txt) | Python dependencies (see setup below) |

Implementation (Open3D pipeline, rendering, experiments) will grow in this repo; see the proposal in `index.html` for milestones.

---

## Prerequisites

- **Python 3.10+** (3.12 is fine). Check with `python --version` or `python3 --version`.
- **Git** and a terminal (**Terminal** on macOS, **PowerShell**, **cmd**, or **Git Bash** on Windows).

On **Windows**, if the project folder lives under **OneDrive**, use the **two-step Open3D install** below. A full `pip install open3d` can hit very long paths inside optional Jupyter packages.

---

## Python setup

Always work from the **repository root** (the folder that contains `requirements.txt`).

### macOS and Linux

Create and activate the venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If `python3` is missing, try `python -m venv .venv`.

### Windows

Create the venv (use `python` if `python3` is not on your PATH):

```bat
python -m venv .venv
```

Activate it (pick the shell you use):

| Shell | Command |
|-------|---------|
| **PowerShell or cmd** | `.venv\Scripts\activate` |
| **Git Bash** | `source .venv/Scripts/activate` |

When activation works, your prompt should show `(.venv)`.

### Install packages

#### 1. Upgrade pip (recommended)

```bash
python -m pip install --upgrade pip
```

#### 2. Install dependencies (two steps)

```bash
pip install -r requirements.txt
pip install "open3d>=0.18.0" --no-deps
```

The second line installs Open3D **without** pulling the full Jupyter widget stack, which avoids **Windows path-length** failures when the repo is under OneDrive.

#### 3. Run scripts with the venv’s Python

If the venv is **activated**, `python` points at `.venv` automatically.

If you **do not** want to activate, call the interpreter explicitly:

| OS | Command prefix |
|----|----------------|
| **macOS / Linux** | `.venv/bin/python` |
| **Windows** | `.venv\Scripts\python.exe` |

Examples (after `cd` to the repo root):

```bash
# macOS / Linux (venv activated)
python data/download_fallback.py
python data/preview.py --source fallback --index 0

# Windows (venv activated — same commands)
python data/download_fallback.py
python data/preview.py --source fallback --index 0

# Windows without activating (PowerShell/cmd from repo root)
.venv\Scripts\python.exe data\preview.py --source fallback --index 0
```

Mesh download and preview details: [`data/README.md`](data/README.md).

---

## Viewing the proposal (`index.html`)

Open in a browser from the repo root:

| OS | Command |
|----|---------|
| **macOS** | `open index.html` |
| **Windows (PowerShell)** | `start index.html` |
| **Windows (cmd)** | `start index.html` |
| **Linux** | `xdg-open index.html` |

Optional local server:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/index.html`.

---

## Planned stack

- **Language:** Python  
- **Geometry / reconstruction:** Open3D (TSDF or voxel fusion, utilities)  
- **Rendering:** Synthetic scenes; Blender or another tool the team standardizes on for RGB/depth from known poses  
- **Hardware:** Team laptops; GPUs as available per member  

## License

TBD by the course team (add a `LICENSE` file when requirements are clear).
