"""Offline simulator that mirrors the Flower aggregation protocol."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from federated_project.dataset import (
    get_num_classes,
    partition_dataset_by_client,
)
from federated_project.federation import (
    ClientUpdate,
    aggregate_client_updates,
    create_model,
    get_client_update_parameters,
    get_global_parameters,
    resolve_device,
    set_global_parameters,
    split_client_update_parameters,
)
from federated_project.train import client_train


@dataclass
class SimulationRoundResult:
    """Simple per-round summary returned by the offline simulator."""

    round_idx: int
    participating_clients: list[int]
    train_loss: float
    spreadout_loss: float


def _sorted_client_names(data_dir: str) -> list[str]:
    """Return stable, human-readable class/client names for a dataset root.

    The simulator expects a multi-person directory layout where each immediate
    subdirectory is a person/class (i.e., a simulated federated client).
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: '{data_dir}'")

    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )


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
) -> list[SimulationRoundResult]:
    """Run the same client/server algorithm locally without Flower networking.

    Defaults follow the FedFace paper (Section 4.1):
      - margin = 0.9
      - lr = 1e-3
      - num_rounds = 200
      - spreadout_strength (lambda) = 10.0
      - one server-side spreadout step per round (steps=1, lr=1.0)
      - full backbone updates (freeze_backbone=False)
    """
    random.seed(seed)
    torch.manual_seed(seed)

    class_names = _sorted_client_names(data_dir)
    num_clients = get_num_classes(data_dir)
    resolved_device = resolve_device(device)
    global_model = create_model(
        num_clients=num_clients,
        pretrained=pretrained,
        device=resolved_device,
        freeze_backbone=freeze_backbone,
    )

    # Two parallel partitionings of the same data:
    #   - train_loaders: with augmentations (used during local training)
    #   - init_loaders : NO augmentations (used for Mean Feature Initialization,
    #     paper Eq. 6, which must operate on the raw client images).
    train_loaders = partition_dataset_by_client(
        data_dir=data_dir,
        train=True,
        batch_size=batch_size,
    )
    init_loaders = partition_dataset_by_client(
        data_dir=data_dir,
        train=False,
        batch_size=batch_size,
    )

    client_ids = sorted(train_loaders.keys())
    sampled_clients = max(1, int(round(len(client_ids) * fraction_fit)))
    sampled_clients = min(len(client_ids), sampled_clients)

    results: list[SimulationRoundResult] = []
    global_parameters = get_global_parameters(global_model)

    for round_idx in range(num_rounds):
        active_clients = sorted(random.sample(client_ids, sampled_clients))
        client_updates: list[ClientUpdate] = []

        for client_id in active_clients:
            local_model = create_model(
                num_clients=num_clients,
                pretrained=pretrained,
                device=resolved_device,
                freeze_backbone=freeze_backbone,
            )
            set_global_parameters(local_model, global_parameters)

            train_metrics = client_train(
                model=local_model,
                dataloader=train_loaders[client_id],
                client_id=client_id,
                round_num=round_idx,
                local_epochs=local_epochs,
                lr=lr,
                margin=margin,
                device=resolved_device,
                # Pass the un-augmented loader for Mean Feature Initialization
                # so that the round-0 anchor (paper Eq. 6) is the true mean
                # of f_theta_0 over the client's raw images.
                init_dataloader=init_loaders[client_id],
            )

            payload = get_client_update_parameters(local_model, client_id)
            backbone_parameters, class_embedding = split_client_update_parameters(
                global_model,
                payload,
            )
            client_updates.append(
                ClientUpdate(
                    client_id=client_id,
                    num_examples=int(train_metrics["num_samples"]),
                    feature_extractor_parameters=backbone_parameters,
                    class_embedding=class_embedding,
                    loss=float(train_metrics["loss"]),
                )
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
        results.append(
            SimulationRoundResult(
                round_idx=round_idx + 1,
                participating_clients=active_clients,
                train_loss=metrics["train_loss"],
                spreadout_loss=metrics["spreadout_loss"],
            )
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
            },
            checkpoint_file,
        )

    return results