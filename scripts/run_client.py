"""CLI entrypoint for starting a Flower client."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from federated_project.client import create_client, start_flower_client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start a Flower face-recognition client.")
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--pretrained", default="vggface2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    client = create_client(
        client_id=args.client_id,
        data_dir=args.data_dir,
        num_clients=args.num_clients,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        lr=args.lr,
        margin=args.margin,
        num_workers=args.num_workers,
        device=args.device,
    )
    start_flower_client(args.server_address, client)


if __name__ == "__main__":
    main()
