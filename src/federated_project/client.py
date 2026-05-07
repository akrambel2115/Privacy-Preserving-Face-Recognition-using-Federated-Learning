"""Flower client implementation for privacy-preserving face recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import flwr as fl
import torch
import torch.nn.functional as F
from flwr.client import NumPyClient
from flwr.clientapp import ClientApp
from flwr.clientapp.mod import LocalDpMod, secaggplus_mod
from flwr.common import Context, NDArrays, Scalar
from torch.utils.data import DataLoader

from federated_project.dataset import load_client_dataset
from federated_project.federation import (
    create_model,
    get_client_update_parameters,
    get_feature_extractor_parameters,
    get_global_parameters,
    get_private_backbone_update_parameters,
    resolve_device,
    set_feature_extractor_parameters,
    set_global_parameters,
)
from federated_project.train import client_train

_LOCAL_CLIENT_EMBEDDINGS: dict[tuple[int, str], torch.Tensor] = {}


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
    backbone weights. In secure mode, the personal class embedding stays local
    and only the shared FaceNet backbone is exchanged.
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
        secure_aggregation: bool = False,
        train_backbone: bool = False,
        preservation_strength: float = 0.0,
        negative_strength: float = 0.0,
        negative_margin: float = 0.2,
    ) -> None:
        self.client_id = client_id
        self.data_dir = data_dir
        self.local_epochs = local_epochs
        self.lr = lr
        self.margin = margin
        self.device = resolve_device(device)
        self.secure_aggregation = secure_aggregation
        self.train_backbone = train_backbone
        self.preservation_strength = preservation_strength
        self.negative_strength = negative_strength
        self.negative_margin = negative_margin

        self.model = create_model(
            num_clients=num_clients,
            pretrained=pretrained,
            device=self.device,
            train_backbone=train_backbone,
        )
        self.reference_model = None
        if train_backbone and preservation_strength > 0.0 and pretrained is not None:
            self.reference_model = create_model(
                num_clients=num_clients,
                pretrained=pretrained,
                device=self.device,
                train_backbone=False,
            )
            self.reference_model.eval()

        embedding_key = self._embedding_key()
        if embedding_key in _LOCAL_CLIENT_EMBEDDINGS:
            self.model.W_matrix.data[client_id] = _LOCAL_CLIENT_EMBEDDINGS[
                embedding_key
            ].to(device=self.device, dtype=self.model.W_matrix.dtype)
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

    def _embedding_key(self) -> tuple[int, str]:
        return self.client_id, str(Path(self.data_dir).resolve())

    def _save_local_embedding(self) -> None:
        _LOCAL_CLIENT_EMBEDDINGS[self._embedding_key()] = (
            self.model.W_matrix[self.client_id].detach().cpu().clone()
        )

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return the client's current local copy of the exchanged model."""
        if self.secure_aggregation or bool(config.get("secure_aggregation", False)):
            return get_feature_extractor_parameters(self.model)
        return get_global_parameters(self.model)

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Run one Flower fit round on the client's private data."""
        server_round = int(config.get("server_round", 1))
        local_epochs = int(config.get("local_epochs", self.local_epochs))
        learning_rate = float(config.get("lr", self.lr))
        margin = float(config.get("margin", self.margin))
        preservation_strength = float(
            config.get("preservation_strength", self.preservation_strength)
        )
        negative_strength = float(config.get("negative_strength", self.negative_strength))
        negative_margin = float(config.get("negative_margin", self.negative_margin))
        secure_aggregation = self.secure_aggregation or bool(
            config.get("secure_aggregation", False)
        )

        if secure_aggregation:
            set_feature_extractor_parameters(self.model, parameters)
        else:
            set_global_parameters(self.model, parameters)

        train_metrics = client_train(
            model=self.model,
            dataloader=self.train_loader,
            client_id=self.client_id,
            round_num=server_round - 1,
            local_epochs=local_epochs,
            lr=learning_rate,
            margin=margin,
            device=self.device,
            reference_model=self.reference_model,
            preservation_strength=preservation_strength,
            negative_strength=negative_strength,
            negative_margin=negative_margin,
        )

        if secure_aggregation:
            self._save_local_embedding()
            updated_parameters = get_private_backbone_update_parameters(
                model=self.model,
            )
            return updated_parameters, 1, {"secure_aggregation": True}

        num_samples = int(train_metrics["num_samples"])
        updated_parameters = get_client_update_parameters(self.model, self.client_id)
        return updated_parameters, num_samples, {
            "client_id": self.client_id,
            "loss": float(train_metrics["loss"]),
            "positive_loss": float(train_metrics["positive_loss"]),
            "preservation_loss": float(train_metrics["preservation_loss"]),
            "negative_loss": float(train_metrics["negative_loss"]),
        }

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate the global model locally with the positive-only objective."""
        if self.secure_aggregation or bool(config.get("secure_aggregation", False)):
            set_feature_extractor_parameters(self.model, parameters)
            return 0.0, 1, {"secure_aggregation": True}

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
        self._save_local_embedding()
        return loss, num_examples, {"client_id": self.client_id}


def create_client(**kwargs: Any) -> FaceFederatedClient:
    """Factory helper used by the CLI scripts."""
    return FaceFederatedClient(**kwargs)


def start_flower_client(server_address: str, client: FaceFederatedClient) -> None:
    """Start the Flower network client."""
    fl.client.start_client(server_address=server_address, client=client.to_client())


def _run_config_value(context: Context, key: str, default: Any) -> Any:
    return context.run_config[key] if key in context.run_config else default


def _run_config_bool(context: Context, key: str, default: bool) -> bool:
    value = _run_config_value(context, key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _local_dp_mod(msg: Any, context: Context, call_next: Any) -> Any:
    """Apply Flower LocalDpMod to secure training replies when enabled."""
    if not _run_config_bool(context, "local-dp-enabled", False):
        return call_next(msg, context)

    local_dp_mod = LocalDpMod(
        clipping_norm=float(_run_config_value(context, "local-dp-clipping-norm", 1.0)),
        sensitivity=float(_run_config_value(context, "local-dp-sensitivity", 1.0)),
        epsilon=float(_run_config_value(context, "local-dp-epsilon", 5.0)),
        delta=float(_run_config_value(context, "local-dp-delta", 1e-5)),
    )
    return local_dp_mod(msg, context, call_next)


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
        secure_aggregation=True,
        train_backbone=_run_config_bool(context, "train-backbone", False),
        preservation_strength=float(_run_config_value(context, "preservation-strength", 0.0)),
        negative_strength=float(_run_config_value(context, "negative-strength", 0.0)),
        negative_margin=float(_run_config_value(context, "negative-margin", 0.2)),
    ).to_client()


app = ClientApp(
    client_fn=client_fn,
    mods=[secaggplus_mod, _local_dp_mod],
)
