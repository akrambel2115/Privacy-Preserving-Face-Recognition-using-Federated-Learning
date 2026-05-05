"""Run many simulation configurations and save the best one."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def _parse_list(raw: str, cast: type) -> list[Any]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one value is required for each parameter list.")
    return [cast(item) for item in values]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid/random search over simulation hyperparameters."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--num-rounds-list", default="3,5,10")
    parser.add_argument("--fraction-fit-list", default="1.0")
    parser.add_argument("--batch-size-list", default="16,32")
    parser.add_argument("--local-epochs-list", default="1,2")
    parser.add_argument("--lr-list", default="0.001,0.0005")
    parser.add_argument("--margin-list", default="0.5")
    parser.add_argument("--pretrained-list", default="vggface2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-dp-enabled", action="store_true")
    parser.add_argument("--local-dp-clipping-norm", type=float, default=1.0)
    parser.add_argument("--local-dp-sensitivity", type=float, default=1.0)
    parser.add_argument("--local-dp-epsilon", type=float, default=5.0)
    parser.add_argument("--local-dp-delta", type=float, default=1e-5)
    parser.add_argument("--no-stream", action="store_true")

    parser.add_argument(
        "--score-metric",
        choices=["final_train_loss", "final_train_plus_spreadout"],
        default="final_train_loss",
        help=(
            "Kept for CLI compatibility. Secure runs do not expose train loss; "
            "completed trials are ranked by duration."
        ),
    )
    parser.add_argument(
        "--search-mode",
        choices=["grid", "random"],
        default="grid",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Required for random mode. Optional cap for grid mode.",
    )
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--output-dir", default="results/tuning")
    return parser


def _score_trial(result: dict[str, Any], score_metric: str) -> float:
    return float(result["duration_sec"])


def _to_python_type(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _run_secure_trial(args: argparse.Namespace, config: dict[str, Any]) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_simulation.py"),
        "--data-dir",
        args.data_dir,
        "--num-rounds",
        str(config["num_rounds"]),
        "--fraction-fit",
        str(config["fraction_fit"]),
        "--batch-size",
        str(config["batch_size"]),
        "--local-epochs",
        str(config["local_epochs"]),
        "--lr",
        str(config["lr"]),
        "--margin",
        str(config["margin"]),
        "--pretrained",
        str(config["pretrained"]),
        "--local-dp-clipping-norm",
        str(args.local_dp_clipping_norm),
        "--local-dp-sensitivity",
        str(args.local_dp_sensitivity),
        "--local-dp-epsilon",
        str(args.local_dp_epsilon),
        "--local-dp-delta",
        str(args.local_dp_delta),
    ]
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.local_dp_enabled:
        command.append("--local-dp-enabled")
    if args.no_stream:
        command.append("--no-stream")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = _build_parser().parse_args()

    grid: dict[str, list[Any]] = {
        "num_rounds": _parse_list(args.num_rounds_list, int),
        "fraction_fit": _parse_list(args.fraction_fit_list, float),
        "batch_size": _parse_list(args.batch_size_list, int),
        "local_epochs": _parse_list(args.local_epochs_list, int),
        "lr": _parse_list(args.lr_list, float),
        "margin": _parse_list(args.margin_list, float),
        "pretrained": _parse_list(args.pretrained_list, str),
    }

    names = list(grid.keys())
    combinations = [dict(zip(names, values)) for values in itertools.product(*(grid[name] for name in names))]

    if args.search_mode == "random":
        if args.max_trials is None or args.max_trials <= 0:
            raise ValueError("--max-trials must be provided and > 0 for random mode.")
        rng = random.Random(args.random_seed)
        trial_count = min(args.max_trials, len(combinations))
        combinations = rng.sample(combinations, k=trial_count)
    elif args.max_trials is not None and args.max_trials > 0:
        combinations = combinations[: args.max_trials]

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_trials_path = output_dir / f"all_trials_{timestamp}.csv"
    best_path = output_dir / f"best_config_{timestamp}.json"

    print(f"Total trials to run: {len(combinations)}")

    all_results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None

    for idx, config in enumerate(combinations, start=1):
        start = time.perf_counter()
        print(f"[{idx}/{len(combinations)}] Running: {config}")
        try:
            _run_secure_trial(args, config)
            duration_sec = time.perf_counter() - start

            result = {
                "status": "ok",
                "duration_sec": round(duration_sec, 3),
                "final_round": int(config["num_rounds"]),
                "final_train_loss": "",
                "final_spreadout_loss": "",
                **config,
            }
            result["score"] = _score_trial(result, args.score_metric)
            all_results.append(result)

            if best_result is None or result["score"] < best_result["score"]:
                best_result = result

            print(
                "    done "
                f"duration_sec={result['duration_sec']:.3f}, "
                f"score={result['score']:.3f}"
            )
        except Exception as exc:  # pragma: no cover - defensive for long tuning jobs
            duration_sec = time.perf_counter() - start
            fail_result = {
                "status": "error",
                "duration_sec": round(duration_sec, 3),
                "error": str(exc),
                **config,
            }
            all_results.append(fail_result)
            print(f"    failed: {exc}")

    if not all_results:
        raise RuntimeError("No trials were run.")

    all_keys = sorted({key for row in all_results for key in row.keys()})
    with all_trials_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in all_results:
            writer.writerow({key: _to_python_type(row.get(key, "")) for key in all_keys})

    if best_result is None:
        raise RuntimeError("All trials failed. Check all_trials CSV for errors.")

    payload = {
        "score_metric": args.score_metric,
        "best": {key: _to_python_type(value) for key, value in best_result.items()},
        "all_trials_csv": str(all_trials_path),
    }
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nBest trial:")
    print(json.dumps(payload["best"], indent=2))
    print(f"Saved all trials to: {all_trials_path}")
    print(f"Saved best config to: {best_path}")


if __name__ == "__main__":
    main()
