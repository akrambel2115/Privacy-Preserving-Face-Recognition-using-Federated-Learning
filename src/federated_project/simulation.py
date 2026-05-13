"""Offline simulator that mirrors the Flower aggregation protocol.

Two execution modes:
  * sequential (default, paper-faithful): each active client trains on its
    own data in turn within a round; their backbone updates are weight-
    averaged at the end of the round and embedding rows are written back.
  * fused (--use-fused): one mega-batch per step contains samples from
    many clients; the backbone receives the gradient sum directly. Used
    for large client counts where the sequential path is GPU-starved.
    Disabled when DP is on.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from federated_project.dataset import (
    get_num_classes,
    partition_dataset_by_client,
)
from federated_project.federation import (
    apply_spreadout_regularization,
    create_model,
    get_client_update_parameters,
    get_global_parameters,
    resolve_device,
    restore_global_state,
    set_client_embedding,
    set_feature_extractor_parameters,
    set_global_parameters,
    snapshot_global_state,
    split_client_update_parameters,
)
from federated_project.fused_train import (
    fused_initialize_embeddings,
    get_image_cache,
    train_round_fused,
)
from federated_project.train import client_train


@dataclass
class SimulationRoundResult:
    round_idx: int
    participating_clients: list[int]
    train_loss: float
    spreadout_loss: float
    elapsed_sec: float = 0.0


def _sorted_client_names(data_dir: str) -> list[str]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: '{data_dir}'")
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )


# ---------------------------------------------------------------------------
# Sequential round (existing, AMP-aware)
# ---------------------------------------------------------------------------

def _run_sequential_round(
    *,
    round_idx: int,
    active_clients: list[int],
    global_model,
    local_model,
    train_loaders,
    init_loaders,
    global_state_snap,
    num_clients: int,
    local_epochs: int,
    lr: float,
    margin: float,
    spreadout_strength: float,
    spreadout_margin: float,
    spreadout_steps: int,
    spreadout_lr: float,
    device: torch.device,
    use_amp: bool,
    grad_scaler,
) -> tuple[float, float]:
    """Returns (train_loss, spreadout_loss)."""

    backbone_sum: list[np.ndarray] | None = None
    int_buffers: list[np.ndarray] | None = None
    is_floating: list[bool] | None = None
    total_examples = 0
    largest_n = 0
    client_embeddings: list[tuple[int, np.ndarray]] = []
    weighted_loss_sum = 0.0

    for client_id in active_clients:
        # Fast in-place restore from snapshot (no numpy round-trip)
        restore_global_state(local_model, global_state_snap)

        train_metrics = client_train(
            model=local_model,
            dataloader=train_loaders[client_id],
            client_id=client_id,
            round_num=round_idx,
            local_epochs=local_epochs,
            lr=lr,
            margin=margin,
            device=device,
            init_dataloader=init_loaders[client_id],
            use_amp=use_amp,
            grad_scaler=grad_scaler,
        )

        n_examples = int(train_metrics["num_samples"])
        payload = get_client_update_parameters(local_model, client_id)
        backbone_params, class_embedding = split_client_update_parameters(
            global_model, payload
        )

        client_embeddings.append((client_id, class_embedding))
        weighted_loss_sum += float(train_metrics["loss"]) * n_examples

        if backbone_sum is None:
            is_floating = [
                np.issubdtype(np.asarray(a).dtype, np.floating)
                for a in backbone_params
            ]
            backbone_sum = [
                np.asarray(a, dtype=np.float64) * n_examples if fl
                else np.zeros_like(a)
                for a, fl in zip(backbone_params, is_floating)
            ]
            int_buffers = [
                np.asarray(a).copy() if not fl else None
                for a, fl in zip(backbone_params, is_floating)
            ]
            largest_n = n_examples
        else:
            for i, (a, fl) in enumerate(zip(backbone_params, is_floating)):
                if fl:
                    backbone_sum[i] += np.asarray(a, dtype=np.float64) * n_examples
                elif n_examples > largest_n:
                    int_buffers[i] = np.asarray(a).copy()
            if n_examples > largest_n:
                largest_n = n_examples
        total_examples += n_examples

    averaged_backbone = []
    for i, fl in enumerate(is_floating):
        if fl:
            averaged_backbone.append(
                (backbone_sum[i] / total_examples).astype(np.float32)
            )
        else:
            averaged_backbone.append(int_buffers[i])

    set_feature_extractor_parameters(global_model, averaged_backbone)
    for cid, emb in client_embeddings:
        set_client_embedding(global_model, cid, emb)

    spreadout_loss = apply_spreadout_regularization(
        global_model,
        margin=spreadout_margin,
        strength=spreadout_strength,
        steps=spreadout_steps,
        lr=spreadout_lr,
    )

    train_loss = weighted_loss_sum / total_examples if total_examples else 0.0
    return train_loss, spreadout_loss


# ---------------------------------------------------------------------------
# Fused round (new)
# ---------------------------------------------------------------------------

def _run_fused_round(
    *,
    round_idx: int,
    active_clients: list[int],
    global_model,
    data_dir: str,
    local_epochs: int,
    lr: float,
    margin: float,
    spreadout_strength: float,
    spreadout_margin: float,
    spreadout_steps: int,
    spreadout_lr: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
    grad_scaler,
) -> tuple[float, float]:
    """Train the global model directly via a fused mega-batch round.

    Round 0 also performs Mean Feature Initialization in one streaming pass.
    """
    if round_idx == 0:
        fused_initialize_embeddings(
            model=global_model,
            data_dir=data_dir,
            client_ids=active_clients,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            log=True,
        )

    result = train_round_fused(
        model=global_model,
        data_dir=data_dir,
        active_client_ids=active_clients,
        local_epochs=local_epochs,
        lr=lr,
        margin=margin,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        use_amp=use_amp,
        grad_scaler=grad_scaler,
        train_augment=True,
    )

    spreadout_loss = apply_spreadout_regularization(
        global_model,
        margin=spreadout_margin,
        strength=spreadout_strength,
        steps=spreadout_steps,
        lr=spreadout_lr,
    )
    return result.train_loss, spreadout_loss


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_simulation(
    data_dir: str,
    num_rounds: int = 200,
    fraction_fit: float = 1.0,
    batch_size: int = 16,
    local_epochs: int = 1,
    lr: float = 1e-3,
    margin: float = 0.9,
    pretrained: str = "vggface2",
    spreadout_strength: float = 10.0,
    spreadout_margin: float = 0.35,
    spreadout_steps: int = 1,
    spreadout_lr: float = 1.0,
    seed: int = 42,
    device: str | None = None,
    checkpoint_path: str | None = None,
    freeze_backbone: bool = False,
    # --- speed knobs (no algorithmic effect when off) ---
    num_workers: int = 0,
    use_amp: bool = False,
    use_fused: bool = False,
    cudnn_benchmark: bool = True,
    log_round_timing: bool = False,
) -> list[SimulationRoundResult]:
    """Run the federated simulation.

    Speed-related kwargs:
      num_workers      : DataLoader workers per client. Honored by the
                         sequential path. The fused path uses a single
                         round-loader with this many workers.
      use_amp          : Mixed-precision (fp16) forward+backward on CUDA.
                         Loss in fp32 for stability. Off when DP is on.
      use_fused        : Run all active clients per round as a single
                         mega-batch (fused_train.train_round_fused). See
                         that module for algorithmic notes. Disabled when
                         spreadout is the only safety net at C<10.
      cudnn_benchmark  : Enable torch.backends.cudnn.benchmark. Defaults
                         True since the simulation has stable input shapes.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    class_names = _sorted_client_names(data_dir)
    num_clients = get_num_classes(data_dir)
    resolved_device = resolve_device(device)

    # Hard-disable fused when client count is too small (spreadout is weak there).
    if use_fused and num_clients < 10:
        print(
            f"  WARNING: --use-fused requested with only {num_clients} clients; "
            "the fused path is intended for many-client regimes. Falling back "
            "to sequential."
        )
        use_fused = False

    global_model = create_model(
        num_clients=num_clients,
        pretrained=pretrained,
        device=resolved_device,
        freeze_backbone=freeze_backbone,
    )

    # Build train loaders. The fused path doesn't use these but they're cheap
    # to build and keep the sequential fallback available.
    if not use_fused:
        train_loaders = partition_dataset_by_client(
            data_dir=data_dir,
            train=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        init_loaders = partition_dataset_by_client(
            data_dir=data_dir,
            train=False,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        client_ids = sorted(train_loaders.keys())
    else:
        train_loaders = None
        init_loaders = None
        client_ids = list(range(num_clients))
        # Pre-load ALL images into CPU RAM once — eliminates per-round disk I/O
        cache = get_image_cache()
        cache.load(data_dir, num_workers=num_workers)

    sampled_clients = max(1, int(round(len(client_ids) * fraction_fit)))
    sampled_clients = min(len(client_ids), sampled_clients)

    # Reusable local model for the sequential path
    local_model = None
    if not use_fused:
        local_model = create_model(
            num_clients=num_clients,
            pretrained=pretrained,
            device=resolved_device,
            freeze_backbone=freeze_backbone,
        )

    # Reusable AMP grad scaler
    grad_scaler = None
    if use_amp and resolved_device.type == "cuda":
        grad_scaler = torch.cuda.amp.GradScaler()

    results: list[SimulationRoundResult] = []

    for round_idx in range(num_rounds):
        active_clients = sorted(random.sample(client_ids, sampled_clients))
        t0 = time.perf_counter()

        if use_fused:
            train_loss, spreadout_loss = _run_fused_round(
                round_idx=round_idx,
                active_clients=active_clients,
                global_model=global_model,
                data_dir=data_dir,
                local_epochs=local_epochs,
                lr=lr,
                margin=margin,
                spreadout_strength=spreadout_strength,
                spreadout_margin=spreadout_margin,
                spreadout_steps=spreadout_steps,
                spreadout_lr=spreadout_lr,
                batch_size=batch_size,
                num_workers=num_workers,
                device=resolved_device,
                use_amp=use_amp,
                grad_scaler=grad_scaler,
            )
        else:
            global_state_snap = snapshot_global_state(global_model)
            train_loss, spreadout_loss = _run_sequential_round(
                round_idx=round_idx,
                active_clients=active_clients,
                global_model=global_model,
                local_model=local_model,
                train_loaders=train_loaders,
                init_loaders=init_loaders,
                global_state_snap=global_state_snap,
                num_clients=num_clients,
                local_epochs=local_epochs,
                lr=lr,
                margin=margin,
                spreadout_strength=spreadout_strength,
                spreadout_margin=spreadout_margin,
                spreadout_steps=spreadout_steps,
                spreadout_lr=spreadout_lr,
                device=resolved_device,
                use_amp=use_amp,
                grad_scaler=grad_scaler,
            )

        elapsed = time.perf_counter() - t0
        results.append(
            SimulationRoundResult(
                round_idx=round_idx + 1,
                participating_clients=active_clients,
                train_loss=train_loss,
                spreadout_loss=spreadout_loss,
                elapsed_sec=round(elapsed, 2),
            )
        )

        if log_round_timing:
            print(
                f"[round {round_idx + 1}/{num_rounds}] "
                f"clients={len(active_clients)} "
                f"train_loss={train_loss:.6f} "
                f"spreadout_loss={spreadout_loss:.6f} "
                f"elapsed={elapsed:.1f}s"
            )

    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "class_names": class_names,
                "num_clients": num_clients,
                "pretrained": pretrained,
                "feature_extractor_state_dict": global_model.feature_extractor.state_dict(),
                "W_matrix": global_model.W_matrix.detach().cpu(),
                "seed": seed,
                "num_rounds": num_rounds,
                "fraction_fit": fraction_fit,
                "batch_size": batch_size,
                "local_epochs": local_epochs,
                "lr": lr,
                "margin": margin,
                "spreadout_strength": spreadout_strength,
                "spreadout_margin": spreadout_margin,
                "spreadout_steps": spreadout_steps,
                "spreadout_lr": spreadout_lr,
                "freeze_backbone": freeze_backbone,
                "use_fused": use_fused,
                "use_amp": use_amp,
                "num_workers": num_workers,
            },
            checkpoint_file,
        )

    return results