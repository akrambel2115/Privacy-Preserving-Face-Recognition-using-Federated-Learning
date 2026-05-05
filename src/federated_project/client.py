"""Flower client implementation for privacy-preserving face recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import flwr as fl
import torch
import torch.nn.functional as F
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context, NDArrays, Scalar
from torch.utils.data import DataLoader

from federated_project.dataset import load_client_dataset
from federated_project.federation import (
    create_model,
    get_client_update_parameters,
    get_global_parameters,
    get_secure_client_update_parameters,
    resolve_device,
    set_global_parameters,
)
from federated_project.train import client_train


@torch.no_grad()
def evaluate_client_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    client_id: int,
    margin: float,
    device: torch.device,
) -> float:
    """Evaluate the positive-only loss on the client's local samples."""
    model.eval()
    total_loss = 0.0
    total_examples = 0

    for images, _ in dataloader:
        images = images.to(device)
        features = model(images)
        class_embedding = F.normalize(model.W_matrix[client_id], p=2, dim=0)
        cosine_similarity = torch.matmul(features, class_embedding)
        hinge = torch.clamp(margin - cosine_similarity, min=0.0)
        batch_loss = (hinge ** 2).mean()

        batch_size = images.size(0)
        total_loss += float(batch_loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


class FaceFederatedClient(NumPyClient):
    """
    Flower NumPyClient for one identity owner.

    Each client keeps its raw images locally, receives the global model from the
    server, performs the local positive-only update, then returns the updated
    backbone weights and only its own class embedding row.
    """

    def __init__(
        self,
        client_id: int,
        data_dir: str,
        num_clients: int,
        pretrained: str = "vggface2",
        batch_size: int = 16,
        local_epochs: int = 1,
        lr: float = 1e-3,
        margin: float = 0.5,
        num_workers: int = 0,
        device: str | None = None,
    ) -> None:
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.lr = lr
        self.margin = margin
        self.device = resolve_device(device)

        self.model = create_model(
            num_clients=num_clients,
            pretrained=pretrained,
            device=self.device,
        )
        self.train_loader = load_client_dataset(
            data_dir=data_dir,
            client_id=client_id,
            train=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.eval_loader = load_client_dataset(
            data_dir=data_dir,
            client_id=client_id,
            train=False,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return the client's current local copy of the global model."""
        return get_global_parameters(self.model)

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Run one Flower fit round on the client's private data."""
        previous_embedding_matrix = parameters[-1].copy()
        set_global_parameters(self.model, parameters)

        server_round = int(config.get("server_round", 1))
        local_epochs = int(config.get("local_epochs", self.local_epochs))
        learning_rate = float(config.get("lr", self.lr))
        margin = float(config.get("margin", self.margin))
        secure_aggregation = bool(config.get("secure_aggregation", False))

        dp_clip_norm = float(config.get("dp_clip_norm", 1.0))
        dp_noise_multiplier = float(config.get("dp_noise_multiplier", 0.0))
        dp_anchor_noise_multiplier = float(config.get("dp_anchor_noise_multiplier", 0.0))
        embedding_init_log_every = int(config.get("embedding_init_log_every", 0))

        if secure_aggregation and (dp_noise_multiplier != 0.0 or dp_anchor_noise_multiplier != 0.0):
            raise NotImplementedError(
                "DP is not yet compatible with the SecAgg+ execution path. "
                "Disable secure_aggregation when enabling DP."
            )

        train_metrics = client_train(
            model=self.model,
            dataloader=self.train_loader,
            client_id=self.client_id,
            round_num=server_round - 1,
            local_epochs=local_epochs,
            lr=learning_rate,
            margin=margin,
            dp_clip_norm=dp_clip_norm,
            dp_noise_multiplier=dp_noise_multiplier,
            device=self.device,
            init_dataloader=self.eval_loader,
            embedding_init_log_every=embedding_init_log_every,
        )

        if secure_aggregation:
            selected_clients = int(config.get("secure_selected_clients", 1))
            updated_parameters = get_secure_client_update_parameters(
                model=self.model,
                client_id=self.client_id,
                selected_clients=selected_clients,
                previous_embedding_matrix=previous_embedding_matrix,
            )
            return updated_parameters, 1, {"secure_aggregation": True}

        num_samples = int(train_metrics["num_samples"])
        updated_parameters = get_client_update_parameters(
            self.model,
            self.client_id,
            n_local_samples=len(self.train_loader.dataset),
            dp_anchor_noise_multiplier=dp_anchor_noise_multiplier,
        )
        return updated_parameters, num_samples, {
            "client_id": self.client_id,
            "loss": float(train_metrics["loss"]),
        }

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate the global model locally with the positive-only objective."""
        set_global_parameters(self.model, parameters)
        margin = float(config.get("margin", self.margin))
        loss = evaluate_client_loss(
            model=self.model,
            dataloader=self.eval_loader,
            client_id=self.client_id,
            margin=margin,
            device=self.device,
        )
        num_examples = len(self.eval_loader.dataset)
        return loss, num_examples, {"client_id": self.client_id}


def create_client(**kwargs: Any) -> FaceFederatedClient:
    """Factory helper used by the CLI scripts."""
    return FaceFederatedClient(**kwargs)


def start_flower_client(server_address: str, client: FaceFederatedClient) -> None:
    """Start the Flower network client."""
    fl.client.start_client(server_address=server_address, client=client.to_client())


def _run_config_value(context: Context, key: str, default: Any) -> Any:
    return context.run_config[key] if key in context.run_config else default


def _resolve_client_data_dir(data_dir: str, client_id: int) -> str:
    """Map a simulation partition id to a person folder when one exists."""
    root = Path(data_dir)
    subdirs = sorted(
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    if subdirs:
        if client_id >= len(subdirs):
            raise ValueError(
                f"client_id {client_id} is outside the available {len(subdirs)} partitions."
            )
        return str(subdirs[client_id])
    return str(root)


def client_fn(context: Context) -> fl.client.Client:
    """Construct a Flower ClientApp client for SecAgg+ runs."""
    client_id = int(
        context.node_config.get(
            "partition-id",
            _run_config_value(context, "client-id", 0),
        )
    )
    num_clients = int(
        _run_config_value(
            context,
            "num-clients",
            context.node_config.get("num-partitions", client_id + 1),
        )
    )
    data_dir = _resolve_client_data_dir(
        str(_run_config_value(context, "data-dir", "data")),
        client_id,
    )

    return create_client(
        client_id=client_id,
        data_dir=data_dir,
        num_clients=num_clients,
        pretrained=str(_run_config_value(context, "pretrained", "vggface2")),
        batch_size=int(_run_config_value(context, "batch-size", 16)),
        local_epochs=int(_run_config_value(context, "local-epochs", 1)),
        lr=float(_run_config_value(context, "learning-rate", 1e-3)),
        margin=float(_run_config_value(context, "margin", 0.5)),
        num_workers=int(_run_config_value(context, "num-workers", 0)),
        device=_run_config_value(context, "device", None),
    ).to_client()


app = ClientApp(
    client_fn=client_fn,
    mods=[secaggplus_mod],
)
