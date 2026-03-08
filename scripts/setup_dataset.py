"""Fix ModelNet directory layout for this project.

Standard ModelNet downloads place class directories directly under
``data/ModelNet10/`` (e.g. ``data/ModelNet10/bed/train/model_001.off``),
but this project's ``DATA_DIR`` / ``MODELNET40_DIR`` point one level deeper::

    data/ModelNet10/models/{class}/{split}/*.off

The extra ``models/`` level keeps class data separate from the point-cloud
cache that is written to ``data/ModelNet10/models/cache/`` at training time.

Run this script once after a fresh ModelNet download to restructure the files
in-place.  It is fully idempotent: running it a second time is safe and
produces only ``[OK]`` / ``[SKIP]`` messages.

Usage::

    python -m scripts.setup_dataset               # fix both datasets (default)
    python -m scripts.setup_dataset --dataset modelnet10
    python -m scripts.setup_dataset --dataset modelnet40
    python -m scripts.setup_dataset --dataset both
"""

import argparse
from pathlib import Path

from src.config import DATA_DIR, MODELNET40_DIR

# Directories that are managed by this project and must not be treated as
# class directories even if they happen to sit at the top level of base_dir.
_PROJECT_DIRS = {"models", "cache"}


def fix_dataset(base_dir: Path) -> None:
    """Move class directories one level down into a ``models/`` subdirectory.

    Args:
        base_dir: Top-level dataset directory, e.g. ``data/ModelNet10/``.
                  The script creates ``base_dir/models/`` and moves every
                  subdirectory that is not in :data:`_PROJECT_DIRS` into it.
    """
    if not base_dir.exists():
        print(f"  [SKIP] {base_dir} — directory not found.")
        return

    class_dirs = sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name not in _PROJECT_DIRS
    )

    if not class_dirs:
        print(f"  [OK]   {base_dir} — already set up correctly.")
        return

    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    for class_dir in class_dirs:
        dest = models_dir / class_dir.name
        if dest.exists():
            print(f"  [SKIP] {class_dir.name}/ — already present in models/")
            continue
        print(f"  Moving  {class_dir.name}/ → models/{class_dir.name}/")
        class_dir.rename(dest)

    print(f"  [DONE] {base_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Move ModelNet class directories into the models/ subdirectory "
            "expected by this project's DATA_DIR / MODELNET40_DIR config."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["modelnet10", "modelnet40", "both"],
        default="both",
        help="Which dataset to fix (default: both).",
    )
    args = parser.parse_args()

    if args.dataset in ("modelnet10", "both"):
        base = DATA_DIR.parent  # data/ModelNet10/
        print(f"\nFixing ModelNet10 at {base} …")
        fix_dataset(base)

    if args.dataset in ("modelnet40", "both"):
        base = MODELNET40_DIR.parent  # data/ModelNet40/
        print(f"\nFixing ModelNet40 at {base} …")
        fix_dataset(base)
