"""Experiment runner: DP tradeoff sweep for FedFace (legacy/NumPyClient path).

This script intentionally does NOT depend on scripts/run_simulation.py.
It drives the same client/server protocol directly in-process and writes
results after each DP setting so partial progress survives crashes.

Outputs:
  results/tradeoff_{timestamp}.csv
  results/tradeoff_{timestamp}.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
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
    set_global_parameters,
    split_client_update_parameters,
)


def _parse_float_list(raw: str) -> list[float]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Noise multiplier list cannot be empty")
    return [float(v) for v in values]


def _sorted_person_dirs(data_dir: Path) -> list[Path]:
    return sorted(
        entry for entry in data_dir.iterdir() if entry.is_dir() and not entry.name.startswith(".")
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "noise_multiplier",
        "epsilon",
        "delta",
        "tar_at_far_0001",
        "clip_norm",
        "num_rounds",
        "timestamp",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_one_training(
    *,
    train_data_dir: Path,
    num_rounds: int,
    fraction_fit: float,
    batch_size: int,
    local_epochs: int,
    lr: float,
    margin: float,
    pretrained: str,
    spreadout_strength: float,
    spreadout_margin: float,
    spreadout_steps: int,
    spreadout_lr: float,
    dp_clip_norm: float,
    dp_noise_multiplier: float,
    dp_anchor_noise_multiplier: float,
    seed: int,
    device: str | None,
    log_every: int,
    client_log_every: int,
    embedding_init_log_every: int,
) -> torch.nn.Module:
    person_dirs = _sorted_person_dirs(train_data_dir)
    if not person_dirs:
        raise FileNotFoundError(f"No person subdirectories under {train_data_dir}")

    num_clients = len(person_dirs)

    rng = random.Random(seed)

    # Global server-side model
    resolved_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = create_model(num_clients=num_clients, pretrained=pretrained, device=resolved_device)
    global_parameters = get_global_parameters(global_model)

    # In-process clients
    clients = [
        FaceFederatedClient(
            client_id=cid,
            data_dir=str(person_dirs[cid]),
            num_clients=num_clients,
            pretrained=pretrained,
            batch_size=batch_size,
            local_epochs=local_epochs,
            lr=lr,
            margin=margin,
            num_workers=0,
            device=str(resolved_device),
        )
        for cid in range(num_clients)
    ]

    all_client_ids = list(range(num_clients))
    sampled_clients = max(1, int(round(num_clients * fraction_fit)))
    sampled_clients = min(num_clients, sampled_clients)

    if log_every <= 0:
        raise ValueError("log_every must be positive")
    if client_log_every < 0:
        raise ValueError("client_log_every must be >= 0")
    if embedding_init_log_every < 0:
        raise ValueError("embedding_init_log_every must be >= 0")

    for round_idx in range(1, num_rounds + 1):
        round_start = time.perf_counter()
        active_ids = sorted(rng.sample(all_client_ids, sampled_clients))

        client_updates: list[ClientUpdate] = []

        for client_idx, cid in enumerate(active_ids, start=1):
            client_start = time.perf_counter()
            updated_payload, num_examples, metrics = clients[cid].fit(
                global_parameters,
                {
                    "server_round": round_idx,
                    "local_epochs": local_epochs,
                    "lr": lr,
                    "margin": margin,
                    "dp_clip_norm": dp_clip_norm,
                    "dp_noise_multiplier": dp_noise_multiplier,
                    "dp_anchor_noise_multiplier": dp_anchor_noise_multiplier,
                    "embedding_init_log_every": embedding_init_log_every,
                    "secure_aggregation": False,
                },
            )

            backbone_params, anchor_row = split_client_update_parameters(global_model, updated_payload)
            client_updates.append(
                ClientUpdate(
                    client_id=int(metrics.get("client_id", cid)),
                    num_examples=int(num_examples),
                    feature_extractor_parameters=backbone_params,
                    class_embedding=anchor_row,
                    loss=float(metrics.get("loss", 0.0)) if metrics else None,
                )
            )

            if client_log_every and (
                client_idx == 1
                or client_idx == len(active_ids)
                or client_idx % client_log_every == 0
            ):
                client_sec = time.perf_counter() - client_start
                loss_val = float(metrics.get("loss", 0.0)) if metrics else 0.0
                print(
                    f"  client={client_idx:>2}/{len(active_ids)} id={cid:>3} "
                    f"loss={loss_val:.6f} sec={client_sec:.2f}"
                )

        metrics = aggregate_client_updates(
            global_model,
            client_updates,
            spreadout_margin=spreadout_margin,
            spreadout_strength=spreadout_strength,
            spreadout_steps=spreadout_steps,
            spreadout_lr=spreadout_lr,
        )
        global_parameters = get_global_parameters(global_model)

        if round_idx % log_every == 0 or round_idx == 1 or round_idx == num_rounds:
            elapsed = time.perf_counter() - round_start
            eta = (num_rounds - round_idx) * elapsed
            print(
                f"round={round_idx:>3}/{num_rounds}  "
                f"clients={len(active_ids):>3}/{num_clients}  "
                f"train_loss={metrics.get('train_loss', 0.0):.6f}  "
                f"spreadout_loss={metrics.get('spreadout_loss', 0.0):.6f}  "
                f"step_sec={elapsed:.2f}  eta_sec={eta:.0f}"
            )

    # Ensure model reflects final aggregated params
    set_global_parameters(global_model, global_parameters)
    return global_model


def main() -> None:
    parser = argparse.ArgumentParser(description="DP tradeoff sweep for FedFace")
    parser.add_argument("--train-data-dir", required=True, help="Root with one subdir per enrolled person")
    parser.add_argument("--pairs-file", required=True, help="Verification pairs file (LFW pairs.txt or simple 3-col format)")
    parser.add_argument("--eval-image-dir", required=True, help="Root directory containing evaluation images")

    parser.add_argument("--noise-multipliers", default="0.0,0.5,1.0,2.0")
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--anchor-noise-multiplier", type=float, default=None, help="Defaults to same as backbone sigma")
    parser.add_argument("--delta", type=float, default=1e-5)

    # Tuned defaults (from results/tuning/sim_tuning_20260428_132443_best.json)
    parser.add_argument("--num-rounds", type=int, default=17)
    parser.add_argument("--fraction-fit", type=float, default=0.9480456499617467)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0038211294416912265)
    parser.add_argument("--margin", type=float, default=0.109732982743667)
    parser.add_argument("--pretrained", default="vggface2")

    parser.add_argument("--spreadout-strength", type=float, default=0.4442156209414605)
    parser.add_argument("--spreadout-margin", type=float, default=0.20787883060031453)
    parser.add_argument("--spreadout-steps", type=int, default=4)
    parser.add_argument("--spreadout-lr", type=float, default=0.5910698619088539)

    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print aggregated train/spreadout loss every N rounds",
    )
    parser.add_argument(
        "--client-log-every",
        type=int,
        default=5,
        help="Print per-client fit progress every N clients (0 disables)",
    )
    parser.add_argument(
        "--embedding-init-log-every",
        type=int,
        default=10,
        help="During round 1 only, print mean-embedding init progress every N batches per client (0 disables)",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    noise_multipliers = _parse_float_list(args.noise_multipliers)
    anchor_sigma_default = args.anchor_noise_multiplier

    timestamp = utc_timestamp().replace(":", "").replace("-", "")
    csv_path = PROJECT_ROOT / "results" / f"tradeoff_{timestamp}.csv"
    json_path = PROJECT_ROOT / "results" / f"tradeoff_{timestamp}.json"

    train_data_dir = Path(args.train_data_dir)
    eval_pairs = load_eval_pairs(args.pairs_file, args.eval_image_dir)

    transform = get_default_transforms(train=False)

    # For simple reporting, compute epsilon against the smallest client dataset.
    person_dirs = _sorted_person_dirs(train_data_dir)
    smallest_client_n = min(sum(1 for _ in d.iterdir() if _.is_file()) for d in person_dirs)

    results_rows: list[dict[str, object]] = []

    experiment_config = {
        "noise_multipliers": noise_multipliers,
        "clip_norm": float(args.clip_norm),
        "num_rounds": int(args.num_rounds),
        "delta": float(args.delta),
        "fraction_fit": float(args.fraction_fit),
        "batch_size": int(args.batch_size),
        "local_epochs": int(args.local_epochs),
        "lr": float(args.lr),
        "margin": float(args.margin),
        "pretrained": str(args.pretrained),
        "spreadout_strength": float(args.spreadout_strength),
        "spreadout_margin": float(args.spreadout_margin),
        "spreadout_steps": int(args.spreadout_steps),
        "spreadout_lr": float(args.spreadout_lr),
        "backbone": "InceptionResnetV1",
        "embedding_dim": 512,
    }

    for sigma in noise_multipliers:
        print(
            "\n" +
            f"=== Running sigma={sigma:.3f} "
            f"(anchor_sigma={'same' if anchor_sigma_default is None else anchor_sigma_default}) ==="
        )
        anchor_sigma = float(anchor_sigma_default) if anchor_sigma_default is not None else float(sigma)

        model = run_one_training(
            train_data_dir=train_data_dir,
            num_rounds=int(args.num_rounds),
            fraction_fit=float(args.fraction_fit),
            batch_size=int(args.batch_size),
            local_epochs=int(args.local_epochs),
            lr=float(args.lr),
            margin=float(args.margin),
            pretrained=str(args.pretrained),
            spreadout_strength=float(args.spreadout_strength),
            spreadout_margin=float(args.spreadout_margin),
            spreadout_steps=int(args.spreadout_steps),
            spreadout_lr=float(args.spreadout_lr),
            dp_clip_norm=float(args.clip_norm),
            dp_noise_multiplier=float(sigma),
            dp_anchor_noise_multiplier=float(anchor_sigma),
            seed=int(args.seed),
            device=args.device,
            log_every=int(args.log_every),
            client_log_every=int(args.client_log_every),
            embedding_init_log_every=int(args.embedding_init_log_every),
        )

        tar = compute_tar_at_far(
            model=model,
            pairs=eval_pairs,
            far_target=0.001,
            device=torch.device(args.device) if args.device else None,
            transform=transform,
        )

        eps_warn = None
        if sigma == 0.0:
            epsilon_csv = float("inf")
            epsilon_json: float | None = None
        else:
            try:
                eps = compute_epsilon(
                    num_rounds=int(args.num_rounds),
                    noise_multiplier=float(sigma),
                    clip_norm=float(args.clip_norm),
                    dataset_size=int(smallest_client_n),
                    batch_size=int(args.batch_size),
                    delta=float(args.delta),
                )
                epsilon_csv = float(eps)
                epsilon_json = float(eps)
            except ImportError:
                # Allow evaluation to complete even if Opacus isn't installed.
                epsilon_csv = float("nan")
                epsilon_json = None
                eps_warn = "epsilon=NA (install opacus)"

        row = {
            "noise_multiplier": float(sigma),
            "epsilon": epsilon_csv,
            "delta": float(args.delta),
            "tar_at_far_0001": float(tar),
            "clip_norm": float(args.clip_norm),
            "num_rounds": int(args.num_rounds),
            "timestamp": utc_timestamp(),
        }
        results_rows.append(row)

        # Persist after each sweep point
        _write_csv(csv_path, results_rows)
        payload = {
            "experiment_config": experiment_config,
            "results": [
                {
                    **{k: v for k, v in r.items() if k != "epsilon"},
                    "epsilon": (None if r["noise_multiplier"] == 0.0 else float(r["epsilon"])),
                }
                for r in results_rows
            ],
        }
        _write_json(json_path, payload)

        epsilon_str = "epsilon=∞" if sigma == 0.0 else (eps_warn or f"epsilon={epsilon_csv:.3f}")
        print(f"sigma={sigma:.3f}  tar@0.1%FAR={tar:.4f}  {epsilon_str}")

    print(f"\nSaved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
