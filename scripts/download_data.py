"""Download the example dataset to a local, ignored directory."""

from __future__ import annotations

from pathlib import Path

import kagglehub


def main() -> None:
    """Download the Celebrity Face Image Dataset outside version control."""
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "celebs"
    path = kagglehub.dataset_download(
        "vishesh1412/celebrity-face-image-dataset",
        output_dir=str(output_dir),
    )
    print(f"Dataset downloaded to: {path}")


if __name__ == "__main__":
    main()
