"""Offline simulator that mirrors the Flower aggregation protocol."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from federated_project.dataset import get_num_classes, partition_dataset_by_client
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


def run_simulation(
    data_dir: str,
    num_rounds: int = 3,
    fraction_fit: float = 1.0,
    batch_size: int = 16,
    local_epochs: int = 1,
    lr: float = 1e-3,
    margin: float = 0.5,
    pretrained: str = "vggface2",
    spreadout_strength: float = 0.0,
    spreadout_margin: float = 0.35,
    spreadout_steps: int = 1,
    spreadout_lr: float = 0.5,
    seed: int = 42,
    device: str | None = None,
    checkpoint_path: str | None = None,
    train_backbone: bool = False,
    preservation_strength: float = 0.0,
    negative_strength: float = 0.0,
    negative_margin: float = 0.2,
) -> list[SimulationRoundResult]:
    """Run the same client/server algorithm locally without Flower networking."""
    random.seed(seed)

    class_names = _sorted_client_names(data_dir)
    num_clients = get_num_classes(data_dir)
    resolved_device = resolve_device(device)
    global_model = create_model(
        num_clients=num_clients,
        pretrained=pretrained,
        device=resolved_device,
        train_backbone=train_backbone,
    )
    client_loaders = partition_dataset_by_client(
        data_dir=data_dir,
        train=True,
        batch_size=batch_size,
    )

    client_ids = sorted(client_loaders.keys())
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
                train_backbone=train_backbone,
            )
            set_global_parameters(local_model, global_parameters)
            reference_model = None
            if train_backbone and preservation_strength > 0.0:
                reference_model = create_model(
                    num_clients=num_clients,
                    pretrained=pretrained,
                    device=resolved_device,
                    train_backbone=False,
                )
                reference_model.eval()

            train_metrics = client_train(
                model=local_model,
                dataloader=client_loaders[client_id],
                client_id=client_id,
                round_num=round_idx,
                local_epochs=local_epochs,
                lr=lr,
                margin=margin,
                device=resolved_device,
                reference_model=reference_model,
                preservation_strength=preservation_strength,
                negative_strength=negative_strength,
                negative_margin=negative_margin,
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

    return results


def _sorted_client_names(data_dir: str) -> list[str]:
    root = Path(data_dir)
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )
