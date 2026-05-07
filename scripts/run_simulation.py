"""CLI entrypoint for running the offline federated simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from federated_project.simulation import run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local federated simulator with FedFace-paper defaults. "
            "Defaults match the paper (Section 4.1): margin=0.9, lr=1e-3, "
            "num_rounds=200, spreadout_strength=10.0, full backbone updates."
        )
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--num-rounds", type=int, default=200)
    parser.add_argument("--fraction-fit", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.9)
    parser.add_argument("--pretrained", default="vggface2")
    parser.add_argument("--spreadout-strength", type=float, default=10.0)
    parser.add_argument("--spreadout-margin", type=float, default=0.35)
    parser.add_argument("--spreadout-steps", type=int, default=1)
    parser.add_argument("--spreadout-lr", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help=(
            "Freeze early backbone layers (legacy behavior). The paper does "
            "not freeze. Off by default."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional path to save final global model checkpoint (.pt).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_simulation(
        data_dir=args.data_dir,
        num_rounds=args.num_rounds,
        fraction_fit=args.fraction_fit,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        lr=args.lr,
        margin=args.margin,
        pretrained=args.pretrained,
        spreadout_strength=args.spreadout_strength,
        spreadout_margin=args.spreadout_margin,
        spreadout_steps=args.spreadout_steps,
        spreadout_lr=args.spreadout_lr,
        seed=args.seed,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        freeze_backbone=args.freeze_backbone,
    )

    for result in results:
        print(
            f"Round {result.round_idx}: clients={result.participating_clients}, "
            f"train_loss={result.train_loss:.6f}, "
            f"spreadout_loss={result.spreadout_loss:.6f}"
        )


if __name__ == "__main__":
    main()