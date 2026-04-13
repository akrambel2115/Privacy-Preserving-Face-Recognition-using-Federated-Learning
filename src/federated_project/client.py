"""Flower client implementation for privacy-preserving face recognition."""

from __future__ import annotations

from typing import Any

import flwr as fl
import torch
import torch.nn.functional as F
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from torch.utils.data import DataLoader

from federated_project.dataset import load_client_dataset
from federated_project.federation import (
    create_model,
    get_client_update_parameters,
    get_global_parameters,
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
        set_global_parameters(self.model, parameters)

        server_round = int(config.get("server_round", 1))
        local_epochs = int(config.get("local_epochs", self.local_epochs))
        learning_rate = float(config.get("lr", self.lr))
        margin = float(config.get("margin", self.margin))

        train_metrics = client_train(
            model=self.model,
            dataloader=self.train_loader,
            client_id=self.client_id,
            round_num=server_round - 1,
            local_epochs=local_epochs,
            lr=learning_rate,
            margin=margin,
            device=self.device,
        )

        updated_parameters = get_client_update_parameters(self.model, self.client_id)
        return updated_parameters, int(train_metrics["num_samples"]), {
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
