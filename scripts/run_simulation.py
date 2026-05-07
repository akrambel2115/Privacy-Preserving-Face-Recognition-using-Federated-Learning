"""CLI entrypoint for running the secure Flower simulation."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _count_clients(data_dir: str) -> int:
    root = Path(data_dir)
    return len(
        [
            path
            for path in root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]
    )


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return repr(value)
    return str(value)


def _run_config(config: dict[str, Any]) -> str:
    return " ".join(f"{key}={_format_value(value)}" for key, value in config.items())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the secure Flower app with SecAgg+ and optional LocalDpMod."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--fraction-fit", type=float, default=1.0)
    parser.add_argument("--min-fit-clients", type=int, default=None)
    parser.add_argument("--min-available-clients", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--pretrained", default="vggface2")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--train-backbone",
        action="store_true",
        help="Fine-tune late FaceNet layers. By default the pretrained backbone is frozen.",
    )
    parser.add_argument(
        "--preservation-strength",
        type=float,
        default=0.0,
        help="Penalty strength for keeping fine-tuned embeddings close to pretrained FaceNet.",
    )
    parser.add_argument(
        "--negative-strength",
        type=float,
        default=0.0,
        help="Penalty strength for pushing images away from other client prototypes.",
    )
    parser.add_argument(
        "--negative-margin",
        type=float,
        default=0.2,
        help="Cosine margin used by the optional prototype-separation penalty.",
    )
    parser.add_argument("--local-dp-enabled", action="store_true")
    parser.add_argument("--local-dp-clipping-norm", type=float, default=1.0)
    parser.add_argument("--local-dp-sensitivity", type=float, default=1.0)
    parser.add_argument("--local-dp-epsilon", type=float, default=5.0)
    parser.add_argument("--local-dp-delta", type=float, default=1e-5)
    parser.add_argument("--num-shares", type=int, default=3)
    parser.add_argument("--reconstruction-threshold", type=int, default=2)
    parser.add_argument("--secure-max-weight", type=float, default=1.0)
    parser.add_argument("--secure-clipping-range", type=float, default=64.0)
    parser.add_argument("--quantization-range", type=int, default=4194304)
    parser.add_argument("--modulus-range", type=int, default=4294967296)
    parser.add_argument("--secure-timeout", type=float, default=60.0)
    parser.add_argument(
        "--federation",
        default=None,
        help="Optional Flower federation name for deployment runtime.",
    )
    parser.add_argument("--no-stream", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    num_clients = args.num_clients or _count_clients(args.data_dir)
    if num_clients <= 0:
        raise ValueError(
            "--num-clients is required when --data-dir has no client subdirectories."
        )

    min_fit_clients = args.min_fit_clients or num_clients
    min_available_clients = args.min_available_clients or num_clients
    config = {
        "data-dir": args.data_dir,
        "num-clients": num_clients,
        "num-server-rounds": args.num_rounds,
        "fraction-fit": args.fraction_fit,
        "min-fit-clients": min_fit_clients,
        "min-available-clients": min_available_clients,
        "batch-size": args.batch_size,
        "local-epochs": args.local_epochs,
        "learning-rate": args.lr,
        "margin": args.margin,
        "pretrained": args.pretrained,
        "num-workers": args.num_workers,
        "train-backbone": args.train_backbone,
        "preservation-strength": args.preservation_strength,
        "negative-strength": args.negative_strength,
        "negative-margin": args.negative_margin,
        "local-dp-enabled": args.local_dp_enabled,
        "local-dp-clipping-norm": args.local_dp_clipping_norm,
        "local-dp-sensitivity": args.local_dp_sensitivity,
        "local-dp-epsilon": args.local_dp_epsilon,
        "local-dp-delta": args.local_dp_delta,
        "num-shares": args.num_shares,
        "reconstruction-threshold": args.reconstruction_threshold,
        "secure-max-weight": args.secure_max_weight,
        "secure-clipping-range": args.secure_clipping_range,
        "quantization-range": args.quantization_range,
        "modulus-range": args.modulus_range,
        "secure-timeout": args.secure_timeout,
    }
    if args.device is not None:
        config["device"] = args.device

    command = ["flwr", "run", "."]
    if args.federation:
        command.append(args.federation)
    if not args.no_stream:
        command.append("--stream")
    command.extend(["--run-config", _run_config(config)])
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
