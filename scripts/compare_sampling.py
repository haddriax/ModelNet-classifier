"""Compare the three point-cloud sampling methods on a single OFF mesh.

Opens a file-picker dialog rooted at ``data/``, loads the selected ``.off``
file, samples **1 024 points** using each of the three available strategies,
then:

1. **Saves** a side-by-side PNG (matplotlib 3-D scatter, fixed isometric view)
   to ``figures/sampling_comparison.png`` at the project root.  The folder is
   created automatically if it does not exist.
2. **Opens** an interactive Open3D window with the three point clouds placed
   side by side and colour-coded by method.

This is designed to generate **Figure 2** of the LaTeX report
(Comparaison des 3 methodes d'echantillonnage).

Usage::

    python -m scripts.compare_sampling
"""

import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must come before pyplot
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from src.builders.mesh_3D_builder import Mesh3DBuilder
from src.config import PROJECT_ROOT
from src.geometry.mesh3d import Mesh3D
from src.geometry.sampling import Sampling

# ── Constants ──────────────────────────────────────────────────────────────────

_DATA_DIR    = PROJECT_ROOT / "data"
_FIGURES_DIR = PROJECT_ROOT / "figures"
_OUTPUT_PNG  = _FIGURES_DIR / "sampling_comparison.png"
_N_POINTS    = 256

_METHODS: list[tuple[str, Sampling]] = [
    ("Uniform",      Sampling.UNIFORM),
    ("FPS",          Sampling.FARTHEST_POINT),
    ("Poisson Disk", Sampling.POISSON),
]

_COLORS: dict[str, str] = {
    "Uniform":      "#3498db",   # blue
    "FPS":          "#e74c3c",   # red
    "Poisson Disk": "#2ecc71",   # green
}

# Matplotlib scatter point size
_SCATTER_S = 1.5

# Open3D window dimensions
_O3D_WIDTH  = 1600
_O3D_HEIGHT = 600


# ── Helpers ────────────────────────────────────────────────────────────────────

def _hex_to_rgb01(hex_color: str) -> list[float]:
    """Convert ``'#rrggbb'`` to an [R, G, B] list in [0, 1]."""
    h = hex_color.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def _pick_off_file() -> Path | None:
    """Open a file-picker dialog and return the selected .off path, or None."""
    initial = _DATA_DIR if _DATA_DIR.exists() else Path.cwd()

    root = tk.Tk()
    root.withdraw()
    raw = filedialog.askopenfilename(
        title="Select an OFF mesh file",
        initialdir=str(initial),
        filetypes=[("OFF files", "*.off"), ("All files", "*.*")],
    )
    root.destroy()

    return Path(raw) if raw else None


# ── PNG figure ─────────────────────────────────────────────────────────────────

def _save_png(clouds: dict[str, np.ndarray], mesh: Mesh3D, output: Path) -> None:
    """Save a 1x3 matplotlib 3-D scatter figure to *output*."""
    output.parent.mkdir(parents=True, exist_ok=True)

    # Pre-build wireframe edge segments once (shared across all sub-plots)
    verts = mesh.vertices                   # Nx3
    faces = mesh.faces                      # Mx3
    # Each face contributes 3 edges; deduplicate by sorting edge endpoints
    edges = set()
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges.add((min(a, b), max(a, b)))
    edges = list(edges)                     # list of (i, j) index pairs

    fig = plt.figure(figsize=(15, 5), dpi=150)

    for idx, (label, pts) in enumerate(clouds.items(), start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")

        # ── Wireframe (drawn first so points sit on top) ──────────────
        xs = np.column_stack([verts[e[0], 0] for e in edges] + [[np.nan]] * len(edges))
        ys = np.column_stack([verts[e[0], 1] for e in edges] + [[np.nan]] * len(edges))
        zs = np.column_stack([verts[e[0], 2] for e in edges] + [[np.nan]] * len(edges))

        # Build interleaved (start, end, nan) arrays for a single plot call
        seg_x = np.empty(len(edges) * 3)
        seg_y = np.empty(len(edges) * 3)
        seg_z = np.empty(len(edges) * 3)
        for k, (i, j) in enumerate(edges):
            seg_x[k * 3]     = verts[i, 0];  seg_x[k * 3 + 1] = verts[j, 0];  seg_x[k * 3 + 2] = np.nan
            seg_y[k * 3]     = verts[i, 1];  seg_y[k * 3 + 1] = verts[j, 1];  seg_y[k * 3 + 2] = np.nan
            seg_z[k * 3]     = verts[i, 2];  seg_z[k * 3 + 1] = verts[j, 2];  seg_z[k * 3 + 2] = np.nan

        ax.plot(seg_x, seg_y, seg_z,
                color="#aaaaaa", linewidth=0.15, alpha=0.35, zorder=1)

        # ── Point cloud ───────────────────────────────────────────────
        color = _COLORS[label]
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=color,
            s=_SCATTER_S,
            alpha=0.85,
            linewidths=0,
            zorder=2,
        )

        ax.view_init(elev=30, azim=45)          # fixed isometric view — same for all 3
        ax.set_title(f"{label}\n{_N_POINTS} pts", fontsize=10, fontweight="bold")
        ax.set_axis_off()

    plt.suptitle("Comparaison des methodes d'echantillonnage", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPNG saved: {output}")


# ── Open3D display ─────────────────────────────────────────────────────────────

def _show_open3d(clouds: dict[str, np.ndarray], mesh: Mesh3D, mesh_name: str) -> None:
    """Display the three point clouds side by side in a single Open3D window."""

    # Compute horizontal offset from the first cloud's bounding box
    first_pts = next(iter(clouds.values()))
    x_range   = float(first_pts[:, 0].max() - first_pts[:, 0].min())
    offset    = x_range * 1.4            # gap between clouds = 40 % of x width

    # Build the shared wireframe once (geometry is the same for every panel)
    base_lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh.triangle_mesh)
    base_lineset.paint_uniform_color([0.65, 0.65, 0.65])   # neutral grey

    geometries: list[o3d.geometry.Geometry] = []

    for i, (label, pts) in enumerate(clouds.items()):
        shift = i * offset

        # Point cloud
        shifted = pts.copy()
        shifted[:, 0] += shift

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(shifted)
        pcd.paint_uniform_color(_hex_to_rgb01(_COLORS[label]))
        geometries.append(pcd)

        # Wireframe (translate the shared lineset vertices)
        wire = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(
                np.asarray(base_lineset.points) + np.array([shift, 0.0, 0.0])
            ),
            lines=base_lineset.lines,
        )
        wire.paint_uniform_color([0.65, 0.65, 0.65])
        geometries.append(wire)

    window_name = (
        f"{mesh_name}  -  Uniform  |  FPS  |  Poisson Disk  -  {_N_POINTS} pts"
    )

    print("Opening Open3D window (close window to exit)...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=_O3D_WIDTH,
        height=_O3D_HEIGHT,
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    off_path = _pick_off_file()
    if off_path is None:
        print("No file selected. Exiting.")
        return

    if not off_path.exists():
        print(f"File not found: {off_path}")
        return

    print(f"\nSampling {_N_POINTS} points from: {off_path.name}")

    mesh   = Mesh3DBuilder.from_off_file(off_path)
    clouds: dict[str, np.ndarray] = {}

    for label, method in _METHODS:
        pts = mesh.sample_points(n_points=_N_POINTS, method=method, force_resample=True)
        clouds[label] = pts
        print(f"  [{label:<12}] -> {len(pts)} pts")

    _save_png(clouds, mesh, _OUTPUT_PNG)
    _show_open3d(clouds, mesh, off_path.name)

    print("\nDone.")


if __name__ == "__main__":
    main()
