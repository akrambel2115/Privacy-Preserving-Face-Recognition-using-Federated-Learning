# train() functions
#
# Local client training for the federated face recognition system.
#
# Each client trains using ONLY positive samples (their own face images).
# The loss is a squared hinge loss with cosine similarity:
#
#     l_pos(f_θ(x), i) = max(0, m − w_i^T · f_θ(x))²
#
# where:
#   f_θ(x) = L2-normalised feature embedding of image x
#   w_i     = L2-normalised class embedding for client i  (row i in W)
#   m       = cosine similarity margin (typically 0.5)
#
# During each communication round the client jointly updates:
#   • θ   — the feature extractor parameters
#   • w_i — its personal class embedding (row i of the global W matrix)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional


# ---------------------------------------------------------------------------
# Feature-extractor mode helpers
# ---------------------------------------------------------------------------

def keep_feature_extractor_eval(model: nn.Module) -> None:
    """Prevent BatchNorm/dropout drift in the pretrained FaceNet backbone."""
    if hasattr(model, "set_feature_extractor_eval"):
        model.set_feature_extractor_eval()
    elif hasattr(model, "feature_extractor"):
        model.feature_extractor.eval()


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def positive_only_loss(
    features: torch.Tensor,
    class_embedding: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Compute the positive-only squared hinge loss with cosine similarity.

    .. math::

        l_{pos} = \\frac{1}{B} \\sum_{j=1}^{B}
                  \\max\\bigl(0,\\; m - {w^{i}}^{\\!T} f_{\\theta}(x_j)\\bigr)^{2}

    Both ``features`` and ``class_embedding`` are expected to be
    L2-normalised so that their dot product equals cosine similarity.

    Args:
        features:        (B, d) tensor of L2-normalised image embeddings
                         produced by the feature extractor ``f_θ(x)``.
        class_embedding: (d,)  tensor — the L2-normalised class embedding
                         ``w_i`` for this client.
        margin:          Cosine-similarity margin ``m``.  The loss is zero
                         when similarity ≥ m for every sample in the batch.

    Returns:
        Scalar loss averaged over the batch.
    """
    # cosine similarity per sample: (B,)
    cos_sim = torch.matmul(features, class_embedding)

    # squared hinge: max(0, m - cos_sim)²
    hinge = torch.clamp(margin - cos_sim, min=0.0)
    loss = (hinge ** 2).mean()
    return loss


def prototype_separation_loss(
    features: torch.Tensor,
    embedding_matrix: torch.Tensor,
    client_id: int,
    margin: float = 0.2,
) -> torch.Tensor:
    """Penalize high similarity to other clients' prototype embeddings."""
    if embedding_matrix.size(0) < 2:
        return features.new_tensor(0.0)

    normalized_embeddings = F.normalize(embedding_matrix, p=2, dim=1)
    negative_mask = torch.ones(
        normalized_embeddings.size(0),
        dtype=torch.bool,
        device=normalized_embeddings.device,
    )
    negative_mask[client_id] = False
    negative_embeddings = normalized_embeddings[negative_mask]
    similarities = features @ negative_embeddings.T
    violations = torch.clamp(similarities - margin, min=0.0)
    return (violations ** 2).mean()


def preservation_loss(
    features: torch.Tensor,
    reference_features: torch.Tensor,
) -> torch.Tensor:
    """Keep fine-tuned embeddings close to the original pretrained space."""
    cosine_similarity = torch.sum(features * reference_features, dim=1)
    return (1.0 - cosine_similarity).mean()


# ---------------------------------------------------------------------------
# Mean Feature Initialization
# ---------------------------------------------------------------------------

@torch.no_grad()
def initialize_client_embedding(
    model: nn.Module,
    dataloader: DataLoader,
    client_id: int,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Initialise a client's class embedding using *Mean Feature Initialization*.

    .. math::

        w_0^{i} = \\frac{1}{n_i} \\sum_{j=1}^{n_i} f_{\\theta_0}(x_j^{i})

    This replaces the random row ``W[client_id]`` with the mean of the
    feature embeddings extracted from the client's local images.  The
    result is then L2-normalised to stay on the unit hypersphere.

    This should be called **once** — in the first communication round —
    before any local training takes place.

    Args:
        model:      The global ``FedFaceModel`` (already on ``device``).
        dataloader: DataLoader over the client's positive-only images.
        client_id:  Index of this client's row in ``model.W_matrix``.
        device:     Torch device to run inference on.
    """
    model.eval()
    keep_feature_extractor_eval(model)

    embedding_sum = None
    num_samples = 0

    for images, _ in dataloader:
        images = images.to(device)
        features = model(images)  # (B, d), already L2-normalised

        if embedding_sum is None:
            embedding_sum = features.sum(dim=0)
        else:
            embedding_sum += features.sum(dim=0)

        num_samples += features.size(0)

    # mean embedding — then L2-normalise
    mean_embedding = embedding_sum / num_samples
    mean_embedding = F.normalize(mean_embedding, p=2, dim=0)

    # overwrite this client's row in the global W matrix
    model.W_matrix.data[client_id] = mean_embedding


# ---------------------------------------------------------------------------
# Single local training round
# ---------------------------------------------------------------------------

def train_one_round(
    model: nn.Module,
    dataloader: DataLoader,
    client_id: int,
    local_epochs: int = 1,
    lr: float = 1e-3,
    margin: float = 0.5,
    device: torch.device = torch.device("cpu"),
    optimizer: Optional[torch.optim.Optimizer] = None,
    reference_model: Optional[nn.Module] = None,
    preservation_strength: float = 0.0,
    negative_strength: float = 0.0,
    negative_margin: float = 0.2,
) -> dict:
    """
    Perform one federated communication round of **local** training.

    The client jointly optimises:
      * ``θ``   — the feature-extractor parameters (unfrozen layers)
      * ``w_i`` — its personal class embedding (``W_matrix[client_id]``)

    using the positive-only squared-hinge loss over ``local_epochs``
    passes of the local dataset.

    Args:
        model:        The global ``FedFaceModel`` (already on ``device``).
        dataloader:   DataLoader over the client's positive-only images.
        client_id:    Row index of this client in ``model.W_matrix``.
        local_epochs: Number of full passes over the local data.
        lr:           Learning rate for the local optimiser.
        margin:       Cosine-similarity margin for the hinge loss.
        device:       Torch device to train on.
        optimizer:    Optional pre-configured optimiser.  If ``None``,
                      an ``Adam`` optimiser is created targeting the
                      trainable feature-extractor params + ``W_matrix``.

    Returns:
        A dict with training statistics::

            {
                "loss":       float,  # average loss over the last epoch
                "num_samples": int,   # total images seen in that epoch
            }
    """
    model.train()
    keep_feature_extractor_eval(model)
    if reference_model is not None:
        reference_model.eval()
        keep_feature_extractor_eval(reference_model)

    # ---- build optimiser if not supplied --------------------------------
    if optimizer is None:
        # Only optimise parameters that require gradients (the unfrozen
        # backbone layers) **plus** the full W_matrix parameter.
        # (W_matrix.requires_grad is True by default since it's nn.Parameter)
        trainable_params = [
            p for p in model.parameters() if p.requires_grad
        ]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)

    # ---- local training loop -------------------------------------------
    epoch_loss = 0.0
    epoch_positive_loss = 0.0
    epoch_preservation_loss = 0.0
    epoch_negative_loss = 0.0
    epoch_samples = 0

    for _epoch in range(local_epochs):
        running_loss = 0.0
        running_positive_loss = 0.0
        running_preservation_loss = 0.0
        running_negative_loss = 0.0
        running_samples = 0

        for images, _ in dataloader:
            images = images.to(device)
            batch_size = images.size(0)

            # 1. Forward: extract L2-normalised features
            features = model(images)  # (B, d)

            # 2. Grab this client's class embedding and L2-normalise it
            #    (we normalise every step so the embedding stays on the
            #    unit hypersphere throughout training)
            w_i = F.normalize(model.W_matrix[client_id], p=2, dim=0)

            # 3. Compute positive-only squared hinge loss
            positive_loss = positive_only_loss(features, w_i, margin=margin)
            loss = positive_loss

            preserve_loss = features.new_tensor(0.0)
            if preservation_strength > 0.0 and reference_model is not None:
                with torch.no_grad():
                    reference_features = reference_model(images)
                preserve_loss = preservation_loss(features, reference_features)
                loss = loss + preservation_strength * preserve_loss

            negative_loss = features.new_tensor(0.0)
            if negative_strength > 0.0:
                negative_loss = prototype_separation_loss(
                    features=features,
                    embedding_matrix=model.W_matrix,
                    client_id=client_id,
                    margin=negative_margin,
                )
                loss = loss + negative_strength * negative_loss

            # 4. Back-propagate and step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            keep_feature_extractor_eval(model)

            running_loss += loss.item() * batch_size
            running_positive_loss += positive_loss.item() * batch_size
            running_preservation_loss += preserve_loss.item() * batch_size
            running_negative_loss += negative_loss.item() * batch_size
            running_samples += batch_size

        epoch_loss = running_loss
        epoch_positive_loss = running_positive_loss
        epoch_preservation_loss = running_preservation_loss
        epoch_negative_loss = running_negative_loss
        epoch_samples = running_samples

    avg_loss = epoch_loss / max(epoch_samples, 1)
    avg_positive_loss = epoch_positive_loss / max(epoch_samples, 1)
    avg_preservation_loss = epoch_preservation_loss / max(epoch_samples, 1)
    avg_negative_loss = epoch_negative_loss / max(epoch_samples, 1)

    return {
        "loss": avg_loss,
        "positive_loss": avg_positive_loss,
        "preservation_loss": avg_preservation_loss,
        "negative_loss": avg_negative_loss,
        "num_samples": epoch_samples,
    }


# ---------------------------------------------------------------------------
# Full client-side routine (init + train)
# ---------------------------------------------------------------------------

def client_train(
    model: nn.Module,
    dataloader: DataLoader,
    client_id: int,
    round_num: int,
    local_epochs: int = 1,
    lr: float = 1e-3,
    margin: float = 0.5,
    device: torch.device = torch.device("cpu"),
    reference_model: Optional[nn.Module] = None,
    preservation_strength: float = 0.0,
    negative_strength: float = 0.0,
    negative_margin: float = 0.2,
) -> dict:
    """
    Complete client-side routine for a single communication round.

    On the **first** round (``round_num == 0``) this function initialises
    the client's class embedding via Mean Feature Initialization before
    training.  On subsequent rounds it jumps straight to local training.

    This is the convenience wrapper that the Flower client should call.

    Args:
        model:        The global ``FedFaceModel`` (already on ``device``).
        dataloader:   DataLoader over the client's positive-only images.
        client_id:    Row index of this client in ``model.W_matrix``.
        round_num:    Current FL communication round (0-indexed).
        local_epochs: Number of local epochs per round.
        lr:           Learning rate for the local optimiser.
        margin:       Cosine-similarity margin for the hinge loss.
        device:       Torch device.

    Returns:
        A dict with training statistics (see :func:`train_one_round`).
    """
    # --- First round: Mean Feature Initialization -----------------------
    if round_num == 0:
        initialize_client_embedding(model, dataloader, client_id, device)

    # --- Local training -------------------------------------------------
    results = train_one_round(
        model=model,
        dataloader=dataloader,
        client_id=client_id,
        local_epochs=local_epochs,
        lr=lr,
        margin=margin,
        device=device,
        reference_model=reference_model,
        preservation_strength=preservation_strength,
        negative_strength=negative_strength,
        negative_margin=negative_margin,
    )

    return results
