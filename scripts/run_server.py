"""CLI entrypoint for starting the Flower aggregation server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from federated_project.server import create_server_strategy, start_flower_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the Flower aggregation server.")
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--pretrained", default="vggface2")
    parser.add_argument("--fraction-fit", type=float, default=1.0)
    parser.add_argument("--min-fit-clients", type=int, default=None)
    parser.add_argument("--min-available-clients", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--spreadout-strength", type=float, default=0.0)
    parser.add_argument("--spreadout-margin", type=float, default=0.35)
    parser.add_argument("--spreadout-steps", type=int, default=1)
    parser.add_argument("--spreadout-lr", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    strategy = create_server_strategy(
        num_clients=args.num_clients,
        pretrained=args.pretrained,
        fraction_fit=args.fraction_fit,
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        margin=args.margin,
        spreadout_strength=args.spreadout_strength,
        spreadout_margin=args.spreadout_margin,
        spreadout_steps=args.spreadout_steps,
        spreadout_lr=args.spreadout_lr,
    )
    start_flower_server(
        server_address=args.server_address,
        num_rounds=args.num_rounds,
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
