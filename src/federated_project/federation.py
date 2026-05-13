"""Shared helpers for Flower-based parameter exchange and aggregation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from federated_project.dp_utils import privatize_anchor
from federated_project.model import FedFaceModel

NDArrays = list[np.ndarray]


@dataclass
class ClientUpdate:
    client_id: int
    num_examples: int
    feature_extractor_parameters: NDArrays
    class_embedding: np.ndarray
    loss: float | None = None


def resolve_device(requested_device: str | None = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(
    num_clients: int,
    pretrained: str = "vggface2",
    device: str | torch.device | None = None,
    freeze_backbone: bool = False,
) -> FedFaceModel:
    resolved_device = device if isinstance(device, torch.device) else resolve_device(device)
    model = FedFaceModel(
        num_clients=num_clients,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
    model.to(resolved_device)
    return model


def _ordered_feature_state(model: FedFaceModel) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(model.feature_extractor.state_dict().items())


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().copy()


def num_feature_tensors(model: FedFaceModel) -> int:
    return len(_ordered_feature_state(model))


def get_feature_extractor_parameters(model: FedFaceModel) -> NDArrays:
    return [_to_numpy(tensor) for tensor in _ordered_feature_state(model).values()]


def set_feature_extractor_parameters(
    model: FedFaceModel,
    parameters: Sequence[np.ndarray],
) -> None:
    """Load serialized backbone tensors into the FaceNet extractor.

    Used by the Flower path where parameters genuinely arrive as numpy.
    The simulation hot path uses ``copy_state_inplace`` (below) instead.
    """
    reference_state = _ordered_feature_state(model)
    if len(parameters) != len(reference_state):
        raise ValueError(
            "Unexpected number of feature tensors. "
            f"Expected {len(reference_state)}, received {len(parameters)}."
        )

    updated_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for (name, reference_tensor), value in zip(reference_state.items(), parameters):
        tensor = torch.from_numpy(np.asarray(value)).to(dtype=reference_tensor.dtype)
        updated_state[name] = tensor

    model.feature_extractor.load_state_dict(updated_state, strict=True)


def get_global_parameters(model: FedFaceModel) -> NDArrays:
    return get_feature_extractor_parameters(model) + [_to_numpy(model.W_matrix)]


def set_global_parameters(
    model: FedFaceModel,
    parameters: Sequence[np.ndarray],
) -> None:
    feature_count = num_feature_tensors(model)
    if len(parameters) != feature_count + 1:
        raise ValueError(
            "Unexpected number of global tensors. "
            f"Expected {feature_count + 1}, received {len(parameters)}."
        )

    set_feature_extractor_parameters(model, parameters[:feature_count])
    embedding_matrix = torch.from_numpy(np.asarray(parameters[-1])).to(
        dtype=model.W_matrix.dtype,
        device=model.W_matrix.device,
    )
    model.W_matrix.data.copy_(F.normalize(embedding_matrix, p=2, dim=1))


# ---------------------------------------------------------------------------
# Fast in-process state copy (simulation hot path)
# ---------------------------------------------------------------------------
#
# The Flower path requires (numpy <-> torch) round-trips because parameters
# are genuinely serialized over the wire. The local simulator does not.
# These helpers move parameters between two FedFaceModel instances, or
# from a snapshot dict, using only torch tensor copies. No numpy involved.
#
# At 1000 clients x 200 rounds = 200,000 calls, the savings are large.

@torch.no_grad()
def snapshot_global_state(model: FedFaceModel) -> dict:
    """Capture the global state as a dict of detached, cloned tensors.

    Stays on the model's device. Used to broadcast the same state to
    every client at the start of a round without re-serialising.
    """
    snap = {
        "feature_extractor": OrderedDict(
            (k, v.detach().clone()) for k, v in model.feature_extractor.state_dict().items()
        ),
        "W_matrix": model.W_matrix.detach().clone(),
    }
    return snap


@torch.no_grad()
def restore_global_state(model: FedFaceModel, snap: dict) -> None:
    """Restore a snapshot in place. Equivalent to set_global_parameters
    but without the numpy round-trip."""
    snap_fe = snap["feature_extractor"]
    # load_state_dict properly writes into the model's actual parameter tensors.
    # Note: state_dict() returns copies, so in-place .copy_() on its output
    # would NOT propagate back to the model — hence we use load_state_dict.
    model.feature_extractor.load_state_dict(snap_fe, strict=True)

    saved_W = snap["W_matrix"].to(device=model.W_matrix.device, dtype=model.W_matrix.dtype)
    model.W_matrix.data.copy_(F.normalize(saved_W, p=2, dim=1))


# ---------------------------------------------------------------------------
# Client payload helpers (Flower path uses these as-is)
# ---------------------------------------------------------------------------

def get_client_update_parameters(
    model: FedFaceModel,
    client_id: int,
    *,
    n_local_samples: int | None = None,
    dp_anchor_noise_multiplier: float = 0.0,
    anchor_sensitivity_override: float | None = None,
) -> NDArrays:
    client_embedding = F.normalize(model.W_matrix[client_id].detach(), p=2, dim=0)
    client_embedding_np = _to_numpy(client_embedding)

    if dp_anchor_noise_multiplier != 0.0:
        if n_local_samples is None:
            raise ValueError("n_local_samples must be provided when anchor DP is enabled")
        client_embedding_np = privatize_anchor(
            anchor=client_embedding_np,
            n_local_samples=int(n_local_samples),
            noise_multiplier=float(dp_anchor_noise_multiplier),
            sensitivity_override=anchor_sensitivity_override,
        )

    return get_feature_extractor_parameters(model) + [client_embedding_np]


def get_secure_client_update_parameters(
    model: FedFaceModel,
    client_id: int,
    selected_clients: int,
    previous_embedding_matrix: np.ndarray,
    *,
    n_local_samples: int | None = None,
    dp_anchor_noise_multiplier: float = 0.0,
    anchor_sensitivity_override: float | None = None,
) -> NDArrays:
    """SecAgg+ payload. See module docstring for the per-row trick."""
    if selected_clients <= 0:
        raise ValueError("selected_clients must be positive.")

    previous_matrix = np.asarray(previous_embedding_matrix).copy()
    if previous_matrix.shape != tuple(model.W_matrix.shape):
        raise ValueError(
            "Unexpected previous embedding matrix shape. "
            f"Expected {tuple(model.W_matrix.shape)}, received {previous_matrix.shape}."
        )

    updated_embedding = _to_numpy(
        F.normalize(model.W_matrix[client_id].detach(), p=2, dim=0)
    )

    if dp_anchor_noise_multiplier != 0.0:
        if n_local_samples is None:
            raise ValueError("n_local_samples must be provided when anchor DP is enabled")
        updated_embedding = privatize_anchor(
            anchor=updated_embedding,
            n_local_samples=int(n_local_samples),
            noise_multiplier=float(dp_anchor_noise_multiplier),
            sensitivity_override=anchor_sensitivity_override,
        )

    previous_row = previous_matrix[client_id].copy()
    previous_matrix[client_id] = previous_row + selected_clients * (
        updated_embedding - previous_row
    )
    return get_feature_extractor_parameters(model) + [previous_matrix]


def split_client_update_parameters(
    model: FedFaceModel,
    parameters: Sequence[np.ndarray],
) -> tuple[NDArrays, np.ndarray]:
    feature_count = num_feature_tensors(model)
    if len(parameters) != feature_count + 1:
        raise ValueError(
            "Unexpected number of client tensors. "
            f"Expected {feature_count + 1}, received {len(parameters)}."
        )
    return list(parameters[:feature_count]), np.asarray(parameters[-1])


def set_client_embedding(model: FedFaceModel, client_id: int, embedding) -> None:
    """Update one row of the global class embedding matrix.

    Accepts either numpy ndarray or torch tensor; avoids needless conversion
    when the source is already torch.
    """
    if isinstance(embedding, torch.Tensor):
        tensor = embedding.to(dtype=model.W_matrix.dtype, device=model.W_matrix.device)
    else:
        tensor = torch.from_numpy(np.asarray(embedding)).to(
            dtype=model.W_matrix.dtype,
            device=model.W_matrix.device,
        )
    model.W_matrix.data[client_id] = F.normalize(tensor, p=2, dim=0)


# ---------------------------------------------------------------------------
# Aggregation helpers (Flower path)
# ---------------------------------------------------------------------------

def weighted_average_ndarrays(
    weighted_parameters: Iterable[tuple[Sequence[np.ndarray], int]],
) -> NDArrays:
    collected = list(weighted_parameters)
    if not collected:
        raise ValueError("Cannot aggregate an empty list of client updates.")

    parameter_sets, example_counts = zip(*collected)
    total_examples = sum(example_counts)
    if total_examples <= 0:
        raise ValueError("The total number of client examples must be positive.")

    aggregated: NDArrays = []
    for arrays_per_tensor in zip(*parameter_sets):
        reference = np.asarray(arrays_per_tensor[0])
        if np.issubdtype(reference.dtype, np.floating):
            weighted_sum = sum(
                np.asarray(array, dtype=np.float64) * num_examples
                for array, num_examples in zip(arrays_per_tensor, example_counts)
            )
            averaged = (weighted_sum / total_examples).astype(reference.dtype)
            aggregated.append(averaged)
        else:
            largest_client_index = int(np.argmax(example_counts))
            aggregated.append(np.asarray(arrays_per_tensor[largest_client_index]).copy())

    return aggregated


# ---------------------------------------------------------------------------
# Spreadout regularizer
# ---------------------------------------------------------------------------

def spreadout_regularization_loss(
    embedding_matrix: torch.Tensor,
    margin: float = 0.35,
) -> torch.Tensor:
    """See docstring in original file. Mean reduction (deliberate)."""
    if embedding_matrix.size(0) < 2:
        return embedding_matrix.new_tensor(0.0)
    normalized = F.normalize(embedding_matrix, p=2, dim=1)
    similarity = normalized @ normalized.T
    mask = ~torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
    violations = F.relu(similarity[mask] - margin)
    if violations.numel() == 0:
        return embedding_matrix.new_tensor(0.0)
    return (violations ** 2).mean()


def apply_spreadout_regularization(
    model: FedFaceModel,
    margin: float = 0.35,
    strength: float = 0.0,
    steps: int = 1,
    lr: float = 0.1,
) -> float:
    if strength <= 0.0 or steps <= 0 or model.W_matrix.size(0) < 2:
        with torch.no_grad():
            model.W_matrix.data.copy_(F.normalize(model.W_matrix.data, p=2, dim=1))
        return 0.0

    optimized = model.W_matrix.detach().clone().requires_grad_(True)
    optimizer = torch.optim.SGD([optimized], lr=lr)
    latest_loss = 0.0

    for _ in range(steps):
        optimizer.zero_grad()
        regularization = strength * spreadout_regularization_loss(optimized, margin=margin)
        latest_loss = float(regularization.item())
        regularization.backward()
        optimizer.step()
        with torch.no_grad():
            optimized.copy_(F.normalize(optimized, p=2, dim=1))

    model.W_matrix.data.copy_(optimized.detach())
    return latest_loss


def aggregate_client_updates(
    model: FedFaceModel,
    client_updates: Sequence[ClientUpdate],
    spreadout_margin: float = 0.35,
    spreadout_strength: float = 0.0,
    spreadout_steps: int = 1,
    spreadout_lr: float = 0.1,
) -> dict[str, float]:
    if not client_updates:
        raise ValueError("At least one client update is required.")

    averaged_backbone = weighted_average_ndarrays(
        [
            (update.feature_extractor_parameters, update.num_examples)
            for update in client_updates
        ]
    )
    set_feature_extractor_parameters(model, averaged_backbone)

    for update in client_updates:
        set_client_embedding(model, update.client_id, update.class_embedding)

    spreadout_loss = apply_spreadout_regularization(
        model,
        margin=spreadout_margin,
        strength=spreadout_strength,
        steps=spreadout_steps,
        lr=spreadout_lr,
    )

    weighted_losses = [
        update.loss * update.num_examples
        for update in client_updates
        if update.loss is not None
    ]
    counted_examples = sum(
        update.num_examples for update in client_updates if update.loss is not None
    )
    train_loss = float(sum(weighted_losses) / counted_examples) if counted_examples else 0.0

    return {
        "train_loss": train_loss,
        "spreadout_loss": spreadout_loss,
    }