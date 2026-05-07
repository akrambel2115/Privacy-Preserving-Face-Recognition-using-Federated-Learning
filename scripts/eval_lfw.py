"""Evaluate a trained checkpoint on an LFW-style pairs file.

Computes TAR @ a target FAR (default 0.1%) using cosine similarity between
the model's L2-normalized embeddings.

This script does NOT train. It only loads a checkpoint and runs
``compute_tar_at_far`` on the verification pairs. It uses the same
preprocessing pipeline as the benchmark suite (no MTCNN), so its numbers
are directly comparable to the ``tar_at_far_0001`` column in
``benchmark_suite_*/summary.csv``.

Example:
  python scripts/eval_lfw.py ^
      --checkpoint results/paper_params/checkpoint.pt ^
      --pairs-file eval/pairs.csv ^
      --eval-image-dir eval/lfw-deepfunneled/lfw-deepfunneled
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from federated_project.dataset import get_default_transforms
from federated_project.dp_utils import compute_tar_at_far, load_eval_pairs
from federated_project.federation import create_model, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute TAR @ FAR on an LFW-style pairs file for a saved checkpoint. "
            "Does not train. Uses the same preprocessing as the benchmark suite."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a benchmark/simulation checkpoint (.pt).",
    )
    parser.add_argument(
        "--pairs-file",
        required=True,
        help="Path to the LFW pairs CSV (e.g. eval/pairs.csv).",
    )
    parser.add_argument(
        "--eval-image-dir",
        required=True,
        help="Path to the LFW image directory with one subfolder per person.",
    )
    parser.add_argument(
        "--far-target",
        type=float,
        default=0.001,
        help="Target false-accept rate. Default 0.001 (= 0.1%%).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cpu or cuda. Defaults to cuda if available.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional path to save the evaluation report as JSON.",
    )
    return parser


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a model from a benchmark/simulation checkpoint dict."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = create_model(
        num_clients=int(ckpt["num_clients"]),
        pretrained=str(ckpt["pretrained"]),
        device=device,
    )
    model.feature_extractor.load_state_dict(
        ckpt["feature_extractor_state_dict"], strict=True
    )

    saved_W = ckpt["W_matrix"].to(device=device, dtype=model.W_matrix.dtype)
    model.W_matrix.data.copy_(F.normalize(saved_W, p=2, dim=1))
    model.eval()
    return model, ckpt


def main() -> None:
    args = build_parser().parse_args()

    checkpoint_path = Path(args.checkpoint)
    pairs_file = Path(args.pairs_file)
    eval_image_dir = Path(args.eval_image_dir)

    for path, label in [
        (checkpoint_path, "Checkpoint"),
        (pairs_file, "Pairs file"),
        (eval_image_dir, "Eval image dir"),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    device = resolve_device(args.device)

    print(f"Loading checkpoint: {checkpoint_path}")
    model, ckpt = load_checkpoint_model(checkpoint_path, device)

    print(f"Loading pairs from : {pairs_file}")
    print(f"Eval image dir     : {eval_image_dir}")
    eval_pairs = load_eval_pairs(str(pairs_file), str(eval_image_dir))

    if not eval_pairs:
        print("ERROR: pairs file produced zero pairs. Check format/paths.")
        sys.exit(1)

    num_genuine = sum(1 for p in eval_pairs if p.is_same_person)
    num_impostor = len(eval_pairs) - num_genuine
    print(f"Loaded {len(eval_pairs)} pairs ({num_genuine} genuine, {num_impostor} impostor)")

    transform = get_default_transforms(train=False)

    print(f"Computing embeddings + TAR @ FAR={args.far_target} ...")
    start = time.perf_counter()
    tar = compute_tar_at_far(
        model=model,
        pairs=eval_pairs,
        far_target=float(args.far_target),
        device=device,
        transform=transform,
    )
    elapsed = time.perf_counter() - start

    far_pct = float(args.far_target) * 100.0
    print()
    print("=" * 60)
    print("  LFW Evaluation Result")
    print("=" * 60)
    print(f"  Checkpoint  : {checkpoint_path}")
    print(f"  Num pairs   : {len(eval_pairs)}  (genuine={num_genuine}, impostor={num_impostor})")
    print(f"  TAR @ {far_pct:.3g}% FAR : {tar:.6f}  ({tar * 100.0:.2f}%)")
    print(f"  Eval time   : {elapsed:.1f}s")
    print("=" * 60)

    if args.report_path:
        report = {
            "checkpoint": str(checkpoint_path),
            "pairs_file": str(pairs_file),
            "eval_image_dir": str(eval_image_dir),
            "far_target": float(args.far_target),
            "tar_at_far": float(tar),
            "num_pairs": len(eval_pairs),
            "num_genuine": num_genuine,
            "num_impostor": num_impostor,
            "elapsed_sec": round(elapsed, 3),
            "checkpoint_metadata": {
                key: (
                    float(value)
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    else (str(value) if not isinstance(value, (list, dict, bool, type(None))) else value)
                )
                for key, value in ckpt.items()
                if key not in {"feature_extractor_state_dict", "W_matrix", "class_names"}
            },
        }
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    main()