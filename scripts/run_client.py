"""CLI entrypoint for starting a secure Flower SuperNode."""

from __future__ import annotations

import argparse
import subprocess


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start a Flower SuperNode for the secure ClientApp."
    )
    parser.add_argument("--superlink", default="127.0.0.1:9092")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument(
        "--clientappio-api-address",
        default=None,
        help=(
            "Optional ClientAppIO address. Set a unique value when running "
            "multiple SuperNodes on one machine."
        ),
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Run without TLS for local development only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    node_config = (
        f"partition-id={args.client_id} "
        f"num-partitions={args.num_clients}"
    )
    command = [
        "flower-supernode",
        "--superlink",
        args.superlink,
        "--node-config",
        node_config,
    ]
    if args.insecure:
        command.append("--insecure")
    if args.clientappio_api_address:
        command.extend(["--clientappio-api-address", args.clientappio_api_address])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
