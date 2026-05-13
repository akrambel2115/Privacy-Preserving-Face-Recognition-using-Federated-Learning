"""Profiling wrapper for the federated simulation.

Runs a short simulation (configurable round count) and reports:
  - Per-round wall-clock timing statistics (mean, std, min, max).
  - GPU memory peak / current (if CUDA is available).
  - Total wall-clock time for the simulation.
  - An optional PyTorch profiler trace (saved to a Chrome-compatible JSON).

Usage examples:

  # Quick 5-round timing sanity check (sequential mode):
  python scripts/profile_simulation.py --data-dir data/train --num-rounds 5

  # Fused mode with AMP, 10 rounds, plus PyTorch profiler trace:
  python scripts/profile_simulation.py --data-dir data/train --num-rounds 10 \
      --use-fused --use-amp --trace-dir ./profiler_traces

  # Compare sequential vs fused:
  python scripts/profile_simulation.py --data-dir data/train --num-rounds 5
  python scripts/profile_simulation.py --data-dir data/train --num-rounds 5 --use-fused
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from federated_project.simulation import run_simulation


# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------

def _gpu_mem_summary() -> dict[str, float] | None:
    """Return GPU memory stats in MB, or None if CUDA unavailable."""
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_MB": torch.cuda.memory_allocated() / (1024 ** 2),
        "reserved_MB": torch.cuda.memory_reserved() / (1024 ** 2),
        "peak_allocated_MB": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "peak_reserved_MB": torch.cuda.max_memory_reserved() / (1024 ** 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Profile the federated simulation for wall-clock & GPU stats.",
    )
    p.add_argument("--data-dir", required=True, help="Path to the training data.")
    p.add_argument("--num-rounds", type=int, default=5,
                    help="Number of federated rounds to run (default 5).")
    p.add_argument("--fraction-fit", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--margin", type=float, default=0.9)
    p.add_argument("--pretrained", default="vggface2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--use-fused", action="store_true")
    p.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark",
                    action="store_false", default=True)

    # Profiler extras
    p.add_argument(
        "--trace-dir", default=None,
        help="If set, run PyTorch profiler and save the Chrome trace JSON here.",
    )
    p.add_argument(
        "--warmup-rounds", type=int, default=1,
        help="Rounds treated as warm-up (excluded from timing stats). Default 1.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    mode = "fused" if args.use_fused else "sequential"
    amp_str = "AMP" if args.use_amp else "fp32"
    print(
        f"\n{'=' * 60}\n"
        f"  Profiling simulation: {args.num_rounds} rounds, mode={mode}, {amp_str}\n"
        f"  data_dir  = {args.data_dir}\n"
        f"  workers   = {args.num_workers}\n"
        f"  cuDNN bench = {args.cudnn_benchmark}\n"
        f"{'=' * 60}\n"
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---- optional PyTorch profiler context ----
    profiler_ctx = None
    if args.trace_dir:
        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        try:
            profiler_ctx = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
            )
        except Exception as exc:
            print(f"  WARNING: Could not create PyTorch profiler: {exc}")
            profiler_ctx = None

    # ---- run simulation ----
    t_start = time.perf_counter()

    if profiler_ctx is not None:
        profiler_ctx.__enter__()

    results = run_simulation(
        data_dir=args.data_dir,
        num_rounds=args.num_rounds,
        fraction_fit=args.fraction_fit,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        lr=args.lr,
        margin=args.margin,
        pretrained=args.pretrained,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
        use_fused=args.use_fused,
        cudnn_benchmark=args.cudnn_benchmark,
        log_round_timing=True,    # always log per-round timing in profiler
    )

    if profiler_ctx is not None:
        profiler_ctx.__exit__(None, None, None)

    t_total = time.perf_counter() - t_start

    # ---- per-round timing stats ----
    all_times = [r.elapsed_sec for r in results]
    warmup = min(args.warmup_rounds, len(all_times))
    measured_times = all_times[warmup:]

    print(f"\n{'─' * 60}")
    print(f"  Total wall-clock time  : {t_total:.2f} s")
    print(f"  Rounds profiled        : {len(measured_times)} (after {warmup} warmup)")
    if measured_times:
        print(f"  Mean round time        : {statistics.mean(measured_times):.3f} s")
        if len(measured_times) > 1:
            print(f"  Std  round time        : {statistics.stdev(measured_times):.3f} s")
        print(f"  Min  round time        : {min(measured_times):.3f} s")
        print(f"  Max  round time        : {max(measured_times):.3f} s")
        estimated_full = statistics.mean(measured_times) * 200
        print(f"  Estimated 200-round    : {estimated_full:.1f} s ({estimated_full / 60:.1f} min)")

    # ---- GPU memory ----
    gpu_mem = _gpu_mem_summary()
    if gpu_mem:
        print(f"\n  GPU peak allocated      : {gpu_mem['peak_allocated_MB']:.1f} MB")
        print(f"  GPU peak reserved       : {gpu_mem['peak_reserved_MB']:.1f} MB")
        print(f"  GPU current allocated   : {gpu_mem['allocated_MB']:.1f} MB")

    if args.trace_dir:
        print(f"\n  Profiler trace saved to : {args.trace_dir}/")

    # ---- per-round table ----
    print(f"\n{'─' * 60}")
    print(f"  {'Round':>6}  {'Loss':>12}  {'Spreadout':>12}  {'Time (s)':>10}")
    print(f"  {'─' * 6}  {'─' * 12}  {'─' * 12}  {'─' * 10}")
    for r in results:
        print(
            f"  {r.round_idx:>6}  {r.train_loss:>12.6f}  "
            f"{r.spreadout_loss:>12.6f}  {r.elapsed_sec:>10.2f}"
        )

    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
