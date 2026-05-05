"""CLI entrypoint for starting the secure Flower SuperLink."""

from __future__ import annotations

import argparse
import subprocess


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Start a Flower SuperLink for the secure ServerApp. Submit the run "
            "with scripts/run_simulation.py or `flwr run .`."
        )
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Run without TLS for local development only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = ["flower-superlink"]
    if args.insecure:
        command.append("--insecure")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
