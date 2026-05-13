r"""Thin wrapper to run the benchmark suite from the repo root.

Usage:
  .\.venv312\Scripts\python.exe benchmark_tradeoff_suite.py <args>

The actual implementation lives in scripts/benchmark_tradeoff_suite.py.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve().parent / "scripts" / "benchmark_tradeoff_suite.py"
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
