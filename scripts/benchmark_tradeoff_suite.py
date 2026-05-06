r"""Benchmark suite: compare DP/SecAgg modes across a sigma sweep.

Runs four modes using the same tuned hyperparameters:
  1) none         : no DP, no secure aggregation (in-process)
  2) ldp          : Local DP only (in-process)
  3) secagg       : SecAgg+ only (Flower App path)
  4) ldp+secagg   : Local DP + SecAgg+ (Flower App path)

For each run, saves a checkpoint (.pt) and computes TAR@FAR on a pairs file.
The suite is crash-safe: it writes results after each completed run.

Example:
  .\.venv312\Scripts\python.exe .\scripts\benchmark_tradeoff_suite.py \
    --train-data-dir "celebs/Celebrity Faces Dataset" \
    --pairs-file "eval/pairs.csv" \
    --eval-image-dir "eval/lfw-deepfunneled/lfw-deepfunneled" \
    --noise-multipliers "0.0,0.5,1.0" \
    --dp-clip-norm 1.0 --delta 1e-5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from federated_project.client import FaceFederatedClient
from federated_project.dataset import get_default_transforms
from federated_project.dp_utils import compute_epsilon, compute_tar_at_far, load_eval_pairs, utc_timestamp
from federated_project.federation import (
    ClientUpdate,
    aggregate_client_updates,
    create_model,
    get_global_parameters,
    resolve_device,
    set_global_parameters,
    split_client_update_parameters,
)


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


@dataclass(frozen=True)
class BestParams:
    num_rounds: int
    fraction_fit: float
    batch_size: int
    local_epochs: int
    lr: float
    margin: float
    pretrained: str
    spreadout_strength: float
    spreadout_margin: float
    spreadout_steps: int
    spreadout_lr: float


def _parse_float_list(raw: str) -> list[float]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Noise multiplier list cannot be empty")
    return [float(v) for v in values]


def _sorted_person_dirs(data_dir: Path) -> list[Path]:
    return sorted(
        entry for entry in data_dir.iterdir() if entry.is_dir() and not entry.name.startswith(".")
    )


def _count_images(directory: Path) -> int:
    return sum(
        1
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def _load_best_params(best_config_path: Path) -> BestParams:
    payload = json.loads(best_config_path.read_text(encoding="utf-8"))

    if "params" in payload:
        params = payload["params"]
    elif "best" in payload:
        params = payload["best"]
    elif "best" in payload.get("best", {}):
        params = payload["best"]["best"]
    else:
        raise ValueError(
            "Unrecognized best-config format. Expected keys like 'params' or 'best'."
        )

    # Support both Optuna and grid-search payload shapes.
    def _get(name: str, default: object | None = None) -> object:
        if name in params:
            return params[name]
        if default is not None:
            return default
        raise KeyError(f"Missing key in best-config: {name}")

    return BestParams(
        num_rounds=int(_get("num_rounds")),
        fraction_fit=float(_get("fraction_fit")),
        batch_size=int(_get("batch_size")),
        local_epochs=int(_get("local_epochs")),
        lr=float(_get("lr")),
        margin=float(_get("margin")),
        pretrained=str(_get("pretrained", "vggface2")),
        spreadout_strength=float(_get("spreadout_strength", 0.0)),
        spreadout_margin=float(_get("spreadout_margin", 0.35)),
        spreadout_steps=int(_get("spreadout_steps", 1)),
        spreadout_lr=float(_get("spreadout_lr", 0.1)),
    )


def _flwr_exe() -> Path:
    """Resolve the Flower CLI executable for the current Python environment."""
    exe_name = "flwr.exe" if sys.platform.startswith("win") else "flwr"
    candidate = Path(sys.executable).with_name(exe_name)
    if candidate.exists():
        return candidate
    # Fall back to PATH lookup (lets users run with global flower installs too).
    return Path(exe_name)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = [
        "mode",
        "sigma",
        "epsilon",
        "delta",
        "tar_at_far_0001",
        "clip_norm",
        "num_rounds",
        "timestamp",
        "checkpoint",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_checkpoint(
    *,
    checkpoint_path: Path,
    model: torch.nn.Module,
    class_names: list[str],
    pretrained: str,
    num_clients: int,
    metadata: dict[str, object],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "class_names": class_names,
            "num_clients": int(num_clients),
            "pretrained": str(pretrained),
            "feature_extractor_state_dict": model.feature_extractor.state_dict(),
            "W_matrix": model.W_matrix.detach().cpu(),
            **metadata,
        },
        checkpoint_path,
    )


def _load_checkpoint_model(checkpoint_path: Path, device: str | None) -> torch.nn.Module:
    resolved_device = resolve_device(device)
    ckpt = torch.load(checkpoint_path, map_location=resolved_device)

    model = create_model(
        num_clients=int(ckpt["num_clients"]),
        pretrained=str(ckpt["pretrained"]),
        device=resolved_device,
    )
    model.feature_extractor.load_state_dict(ckpt["feature_extractor_state_dict"], strict=True)

    saved_W = ckpt["W_matrix"].to(device=resolved_device, dtype=model.W_matrix.dtype)
    model.W_matrix.data.copy_(F.normalize(saved_W, p=2, dim=1))
    model.eval()
    return model


def _run_one_training_inprocess(
    *,
    train_data_dir: Path,
    best: BestParams,
    dp_clip_norm: float,
    dp_noise_multiplier: float,
    dp_anchor_noise_multiplier: float,
    seed: int,
    device: str | None,
) -> torch.nn.Module:
    person_dirs = _sorted_person_dirs(train_data_dir)
    if not person_dirs:
        raise FileNotFoundError(f"No person subdirectories under {train_data_dir}")

    num_clients = len(person_dirs)
    rng = random.Random(seed)

    resolved_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = create_model(num_clients=num_clients, pretrained=best.pretrained, device=resolved_device)
    global_parameters = get_global_parameters(global_model)

    clients = [
        FaceFederatedClient(
            client_id=cid,
            data_dir=str(person_dirs[cid]),
            num_clients=num_clients,
            pretrained=best.pretrained,
            batch_size=best.batch_size,
            local_epochs=best.local_epochs,
            lr=best.lr,
            margin=best.margin,
            num_workers=0,
            device=str(resolved_device),
        )
        for cid in range(num_clients)
    ]

    all_client_ids = list(range(num_clients))
    sampled_clients = max(1, int(round(num_clients * best.fraction_fit)))
    sampled_clients = min(num_clients, sampled_clients)

    for round_idx in range(1, best.num_rounds + 1):
        active_ids = sorted(rng.sample(all_client_ids, sampled_clients))
        client_updates: list[ClientUpdate] = []

        for cid in active_ids:
            updated_payload, num_examples, metrics = clients[cid].fit(
                global_parameters,
                {
                    "server_round": round_idx,
                    "local_epochs": best.local_epochs,
                    "lr": best.lr,
                    "margin": best.margin,
                    "dp_clip_norm": dp_clip_norm,
                    "dp_noise_multiplier": dp_noise_multiplier,
                    "dp_anchor_noise_multiplier": dp_anchor_noise_multiplier,
                    "secure_aggregation": False,
                },
            )

            backbone_params, anchor_row = split_client_update_parameters(global_model, updated_payload)
            client_updates.append(
                ClientUpdate(
                    client_id=int(metrics.get("client_id", cid)) if metrics else int(cid),
                    num_examples=int(num_examples),
                    feature_extractor_parameters=backbone_params,
                    class_embedding=anchor_row,
                    loss=float(metrics.get("loss", 0.0)) if metrics else None,
                )
            )

        metrics = aggregate_client_updates(
            global_model,
            client_updates,
            spreadout_margin=best.spreadout_margin,
            spreadout_strength=best.spreadout_strength,
            spreadout_steps=best.spreadout_steps,
            spreadout_lr=best.spreadout_lr,
        )
        global_parameters = get_global_parameters(global_model)

        if round_idx == 1 or round_idx == best.num_rounds or round_idx % 5 == 0:
            print(
                f"round={round_idx:>3}/{best.num_rounds} "
                f"clients={len(active_ids):>3}/{num_clients} "
                f"train_loss={metrics.get('train_loss', 0.0):.6f} "
                f"spreadout_loss={metrics.get('spreadout_loss', 0.0):.6f}"
            )

    set_global_parameters(global_model, global_parameters)
    return global_model


def _run_secagg_training(
    *,
    train_data_dir: Path,
    best: BestParams,
    dp_clip_norm: float,
    dp_noise_multiplier: float,
    dp_anchor_noise_multiplier: float,
    seed: int,
    device: str | None,
    checkpoint_path: Path,
    log_path: Path,
) -> None:
    person_dirs = _sorted_person_dirs(train_data_dir)
    if not person_dirs:
        raise FileNotFoundError(f"No person subdirectories under {train_data_dir}")
    num_clients = len(person_dirs)

    # Build a single run-config override string. Keys must exist in pyproject.toml.
    data_dir_value = train_data_dir.resolve().as_posix()
    checkpoint_value = checkpoint_path.resolve().as_posix()

    run_config = (
        f"num-clients={num_clients} "
        f"num-server-rounds={best.num_rounds} "
        f"data-dir=\"{data_dir_value}\" "
        f"pretrained=\"{best.pretrained}\" "
        f"fraction-fit={best.fraction_fit} "
        f"min-fit-clients={num_clients} "
        f"min-available-clients={num_clients} "
        f"local-epochs={best.local_epochs} "
        f"learning-rate={best.lr} "
        f"margin={best.margin} "
        f"batch-size={best.batch_size} "
        f"spreadout-strength={best.spreadout_strength} "
        f"spreadout-margin={best.spreadout_margin} "
        f"spreadout-steps={best.spreadout_steps} "
        f"spreadout-lr={best.spreadout_lr} "
        f"dp-clip-norm={dp_clip_norm} "
        f"dp-noise-multiplier={dp_noise_multiplier} "
        f"dp-anchor-noise-multiplier={dp_anchor_noise_multiplier} "
        f"checkpoint-path=\"{checkpoint_value}\""
    )

    if device:
        run_config += f" device=\"{device}\""

    cmd = [
        str(_flwr_exe()),
        "run",
        str(PROJECT_ROOT),
        "--stream",
        "--run-config",
        run_config,
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n")
        log_file.write(" ".join(cmd) + "\n\n")
        log_file.flush()

        start = time.perf_counter()
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        elapsed = time.perf_counter() - start

    if proc.returncode != 0:
        raise RuntimeError(f"SecAgg run failed (exit={proc.returncode}). See: {log_path}")
    if not checkpoint_path.exists():
        raise RuntimeError(
            "SecAgg run completed but did not produce checkpoint. "
            "Ensure server run_config includes checkpoint-path."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark suite: DP/SecAgg tradeoff")
    parser.add_argument("--train-data-dir", required=True)
    parser.add_argument("--pairs-file", required=True)
    parser.add_argument("--eval-image-dir", required=True)

    parser.add_argument(
        "--best-config",
        default=str(PROJECT_ROOT / "results" / "tuning" / "sim_tuning_20260428_132443_best.json"),
        help="Path to best tuned config JSON (Optuna or grid-search).",
    )

    parser.add_argument("--noise-multipliers", default="0.0,0.5,1.0,2.0")
    parser.add_argument("--dp-clip-norm", type=float, default=1.0)
    parser.add_argument("--anchor-noise-multiplier", type=float, default=None)
    parser.add_argument("--delta", type=float, default=1e-5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    best = _load_best_params(Path(args.best_config))
    noise_multipliers = _parse_float_list(args.noise_multipliers)
    anchor_sigma_default = args.anchor_noise_multiplier

    train_data_dir = Path(args.train_data_dir)
    person_dirs = _sorted_person_dirs(train_data_dir)
    if not person_dirs:
        raise FileNotFoundError(f"No person subdirectories under {train_data_dir}")
    num_clients = len(person_dirs)
    class_names = [d.name for d in person_dirs]

    smallest_client_n = min(_count_images(d) for d in person_dirs)

    run_id = utc_timestamp().replace(":", "").replace("-", "")
    out_dir = PROJECT_ROOT / "results" / f"benchmark_suite_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_pairs = load_eval_pairs(args.pairs_file, args.eval_image_dir)
    transform = get_default_transforms(train=False)

    experiment_config = {
        "best_config": str(Path(args.best_config).resolve()),
        "best_params": {
            "num_rounds": best.num_rounds,
            "fraction_fit": best.fraction_fit,
            "batch_size": best.batch_size,
            "local_epochs": best.local_epochs,
            "lr": best.lr,
            "margin": best.margin,
            "pretrained": best.pretrained,
            "spreadout_strength": best.spreadout_strength,
            "spreadout_margin": best.spreadout_margin,
            "spreadout_steps": best.spreadout_steps,
            "spreadout_lr": best.spreadout_lr,
        },
        "noise_multipliers": noise_multipliers,
        "dp_clip_norm": float(args.dp_clip_norm),
        "delta": float(args.delta),
        "anchor_noise_multiplier": (
            None if anchor_sigma_default is None else float(anchor_sigma_default)
        ),
        "num_clients": int(num_clients),
    }

    summary_rows: list[dict[str, object]] = []
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"

    def persist() -> None:
        _write_csv(summary_csv, summary_rows)

        json_results: list[dict[str, object]] = []
        for row in summary_rows:
            epsilon_val = row.get("epsilon")
            sigma_val = float(row.get("sigma", 0.0))
            if sigma_val == 0.0:
                epsilon_json = None
            elif isinstance(epsilon_val, (int, float)) and math.isfinite(float(epsilon_val)):
                epsilon_json = float(epsilon_val)
            else:
                epsilon_json = None

            json_results.append({**row, "epsilon": epsilon_json})

        _write_json(
            summary_json,
            {
                "experiment_config": experiment_config,
                "results": json_results,
            },
        )

    # Baselines (sigma=0) so we don't re-run them for every sigma.
    baseline_runs = [
        ("none", 0.0),
        ("secagg", 0.0),
    ]

    baseline_checkpoints: dict[str, Path] = {}

    for mode, sigma in baseline_runs:
        print(f"\n=== Mode={mode} sigma={sigma:.3f} (baseline) ===")
        anchor_sigma = float(anchor_sigma_default) if anchor_sigma_default is not None else float(sigma)

        mode_dir = out_dir / mode / f"sigma_{sigma:.3f}".replace(".", "p")
        ckpt_path = mode_dir / "checkpoint.pt"

        start = time.perf_counter()
        if mode == "none":
            model = _run_one_training_inprocess(
                train_data_dir=train_data_dir,
                best=best,
                dp_clip_norm=float(args.dp_clip_norm),
                dp_noise_multiplier=0.0,
                dp_anchor_noise_multiplier=0.0,
                seed=int(args.seed),
                device=args.device,
            )
            _save_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                class_names=class_names,
                pretrained=best.pretrained,
                num_clients=num_clients,
                metadata={
                    "mode": mode,
                    "secure_aggregation": False,
                    "dp_noise_multiplier": 0.0,
                    "dp_anchor_noise_multiplier": 0.0,
                    "dp_clip_norm": float(args.dp_clip_norm),
                    "delta": float(args.delta),
                },
            )
        elif mode == "secagg":
            log_path = mode_dir / "flwr_run.log"
            _run_secagg_training(
                train_data_dir=train_data_dir,
                best=best,
                dp_clip_norm=float(args.dp_clip_norm),
                dp_noise_multiplier=0.0,
                dp_anchor_noise_multiplier=0.0,
                seed=int(args.seed),
                device=args.device,
                checkpoint_path=ckpt_path,
                log_path=log_path,
            )
            model = _load_checkpoint_model(ckpt_path, device=args.device)
        else:
            raise AssertionError("Unexpected baseline mode")

        tar = compute_tar_at_far(
            model=model,
            pairs=eval_pairs,
            far_target=0.001,
            device=torch.device(args.device) if args.device else None,
            transform=transform,
        )

        elapsed = time.perf_counter() - start
        row = {
            "mode": mode,
            "sigma": float(sigma),
            "epsilon": float("inf") if sigma == 0.0 else float("nan"),
            "delta": float(args.delta),
            "tar_at_far_0001": float(tar),
            "clip_norm": float(args.dp_clip_norm),
            "num_rounds": int(best.num_rounds),
            "timestamp": utc_timestamp(),
            "checkpoint": str(ckpt_path),
            "runtime_sec": round(elapsed, 3),
        }
        summary_rows.append(row)
        persist()
        print(f"{mode}: tar@0.1%FAR={tar:.4f}  sec={elapsed:.1f}")

        baseline_checkpoints[mode] = ckpt_path

    # Sigma sweep for DP modes (ldp and ldp+secagg)
    for sigma in noise_multipliers:
        anchor_sigma = float(anchor_sigma_default) if anchor_sigma_default is not None else float(sigma)

        for mode in ("ldp", "ldp+secagg"):
            print(f"\n=== Mode={mode} sigma={sigma:.3f} ===")
            mode_dir = out_dir / mode / f"sigma_{sigma:.3f}".replace(".", "p")
            ckpt_path = mode_dir / "checkpoint.pt"

            start = time.perf_counter()
            if sigma == 0.0:
                # Avoid redundant training: copy baseline artifacts while still
                # producing per-mode/per-sigma checkpoint outputs.
                source = baseline_checkpoints["none"] if mode == "ldp" else baseline_checkpoints["secagg"]
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, ckpt_path)
                model = _load_checkpoint_model(ckpt_path, device=args.device)
            elif mode == "ldp":
                model = _run_one_training_inprocess(
                    train_data_dir=train_data_dir,
                    best=best,
                    dp_clip_norm=float(args.dp_clip_norm),
                    dp_noise_multiplier=float(sigma),
                    dp_anchor_noise_multiplier=float(anchor_sigma),
                    seed=int(args.seed),
                    device=args.device,
                )
                _save_checkpoint(
                    checkpoint_path=ckpt_path,
                    model=model,
                    class_names=class_names,
                    pretrained=best.pretrained,
                    num_clients=num_clients,
                    metadata={
                        "mode": mode,
                        "secure_aggregation": False,
                        "dp_noise_multiplier": float(sigma),
                        "dp_anchor_noise_multiplier": float(anchor_sigma),
                        "dp_clip_norm": float(args.dp_clip_norm),
                        "delta": float(args.delta),
                    },
                )
            else:
                log_path = mode_dir / "flwr_run.log"
                _run_secagg_training(
                    train_data_dir=train_data_dir,
                    best=best,
                    dp_clip_norm=float(args.dp_clip_norm),
                    dp_noise_multiplier=float(sigma),
                    dp_anchor_noise_multiplier=float(anchor_sigma),
                    seed=int(args.seed),
                    device=args.device,
                    checkpoint_path=ckpt_path,
                    log_path=log_path,
                )
                model = _load_checkpoint_model(ckpt_path, device=args.device)

            tar = compute_tar_at_far(
                model=model,
                pairs=eval_pairs,
                far_target=0.001,
                device=torch.device(args.device) if args.device else None,
                transform=transform,
            )

            if sigma == 0.0:
                epsilon_csv = float("inf")
                epsilon_json: float | None = None
                eps_warn = None
            else:
                try:
                    eps = compute_epsilon(
                        num_rounds=int(best.num_rounds),
                        noise_multiplier=float(sigma),
                        clip_norm=float(args.dp_clip_norm),
                        dataset_size=int(smallest_client_n),
                        batch_size=int(best.batch_size),
                        delta=float(args.delta),
                    )
                    epsilon_csv = float(eps)
                    epsilon_json = float(eps)
                    eps_warn = None
                except ImportError:
                    epsilon_csv = float("nan")
                    epsilon_json = None
                    eps_warn = "epsilon=NA (install opacus)"

            elapsed = time.perf_counter() - start
            row = {
                "mode": mode,
                "sigma": float(sigma),
                "epsilon": epsilon_csv,
                "delta": float(args.delta),
                "tar_at_far_0001": float(tar),
                "clip_norm": float(args.dp_clip_norm),
                "num_rounds": int(best.num_rounds),
                "timestamp": utc_timestamp(),
                "checkpoint": str(ckpt_path),
                "runtime_sec": round(elapsed, 3),
            }


            summary_rows.append(row)
            persist()

            epsilon_str = "epsilon=∞" if sigma == 0.0 else (eps_warn or f"epsilon={epsilon_csv:.3f}")
            print(f"{mode}: sigma={sigma:.3f}  tar@0.1%FAR={tar:.4f}  {epsilon_str}  sec={elapsed:.1f}")

    print(f"\nSaved suite results to: {out_dir}")


if __name__ == "__main__":
    main()
