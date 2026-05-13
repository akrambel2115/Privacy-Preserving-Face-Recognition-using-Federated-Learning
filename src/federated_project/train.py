# train() functions
#
# Local client training for the federated face recognition system.
#
# Each client trains using ONLY positive samples (their own face images).
# Loss = squared-hinge cosine: max(0, m - w_i^T f_theta(x))^2

import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from federated_project.dp_utils import add_gaussian_noise_inplace, compute_clipped_grad_sum


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def positive_only_loss(
    features: torch.Tensor,
    class_embedding: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    cos_sim = torch.matmul(features, class_embedding)
    hinge = torch.clamp(margin - cos_sim, min=0.0)
    return (hinge ** 2).mean()


# ---------------------------------------------------------------------------
# Mean Feature Initialization
# ---------------------------------------------------------------------------

@torch.no_grad()
def initialize_client_embedding(
    model: nn.Module,
    dataloader: DataLoader,
    client_id: int,
    device: torch.device = torch.device("cpu"),
    log_every_batches: int = 0,
    log_prefix: str = "",
) -> None:
    """Mean Feature Initialization (paper Eq. 6)."""
    model.eval()
    if log_every_batches < 0:
        raise ValueError("log_every_batches must be >= 0")

    embedding_sum = None
    num_samples = 0
    t0 = time.perf_counter()

    if log_every_batches:
        try:
            total_batches = len(dataloader)
        except TypeError:
            total_batches = None

    for batch_idx, (images, _) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        features = model(images)

        if embedding_sum is None:
            embedding_sum = features.sum(dim=0)
        else:
            embedding_sum += features.sum(dim=0)

        num_samples += features.size(0)

        if log_every_batches and (batch_idx == 1 or batch_idx % log_every_batches == 0):
            elapsed = time.perf_counter() - t0
            if total_batches:
                print(
                    f"{log_prefix}init_embed client={client_id} "
                    f"batch={batch_idx}/{total_batches} samples={num_samples} sec={elapsed:.2f}"
                )
            else:
                print(
                    f"{log_prefix}init_embed client={client_id} "
                    f"batch={batch_idx} samples={num_samples} sec={elapsed:.2f}"
                )

    mean_embedding = embedding_sum / num_samples
    mean_embedding = F.normalize(mean_embedding, p=2, dim=0)
    model.W_matrix.data[client_id] = mean_embedding


# ---------------------------------------------------------------------------
# Local training round (sequential, per-client)
# ---------------------------------------------------------------------------

def train_one_round(
    model: nn.Module,
    dataloader: DataLoader,
    client_id: int,
    local_epochs: int = 1,
    lr: float = 1e-3,
    margin: float = 0.5,
    dp_clip_norm: float = 1.0,
    dp_noise_multiplier: float = 0.0,
    device: torch.device = torch.device("cpu"),
    optimizer: Optional[torch.optim.Optimizer] = None,
    use_amp: bool = False,
    grad_scaler: Optional["torch.cuda.amp.GradScaler"] = None,
) -> dict:
    """One federated communication round of local training.

    Args (additions vs paper-faithful version):
        use_amp:     Enable mixed-precision (fp16 forward+backward) on CUDA.
                     The hinge loss is computed in fp32 to avoid underflow
                     near the margin. Has no effect when device is CPU.
                     Disabled automatically when DP is on (per-sample
                     gradient clipping is incompatible with AMP scaling).
        grad_scaler: Optional pre-built GradScaler. Created on demand if
                     use_amp=True and grad_scaler is None.
    """
    model.train()

    if optimizer is None:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)

    # AMP is incompatible with our DP path (per-sample grads via autograd.grad
    # don't compose with GradScaler scaling).
    amp_active = bool(use_amp) and device.type == "cuda" and dp_noise_multiplier == 0.0
    if amp_active and grad_scaler is None:
        grad_scaler = torch.cuda.amp.GradScaler()

    epoch_loss = 0.0
    epoch_samples = 0

    for _epoch in range(local_epochs):
        running_loss = 0.0
        running_samples = 0

        for images, _ in dataloader:
            images = images.to(device, non_blocking=True)
            batch_size = images.size(0)

            optimizer.zero_grad(set_to_none=True)

            if dp_noise_multiplier == 0.0:
                # Standard path (DP off)
                if amp_active:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        features = model(images)
                    # Loss in fp32: features cast back, anchor in fp32.
                    features_fp32 = features.float()
                    w_i = F.normalize(model.W_matrix[client_id], p=2, dim=0)
                    cos_sim = torch.matmul(features_fp32, w_i)
                    hinge = torch.clamp(margin - cos_sim, min=0.0)
                    loss = (hinge ** 2).mean()
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    features = model(images)
                    w_i = F.normalize(model.W_matrix[client_id], p=2, dim=0)
                    cos_sim = torch.matmul(features, w_i)
                    hinge = torch.clamp(margin - cos_sim, min=0.0)
                    loss = (hinge ** 2).mean()
                    loss.backward()
                    optimizer.step()
            else:
                # DP path (unchanged, no AMP)
                features = model(images)
                w_i = F.normalize(model.W_matrix[client_id], p=2, dim=0)
                cos_sim = torch.matmul(features, w_i)
                hinge = torch.clamp(margin - cos_sim, min=0.0)
                per_sample_losses = hinge ** 2
                loss = per_sample_losses.mean()
                loss.backward(retain_graph=True)

                backbone_params = [
                    p for p in model.feature_extractor.parameters() if p.requires_grad
                ]
                clipped_sums = compute_clipped_grad_sum(
                    per_sample_losses, backbone_params, clip_norm=dp_clip_norm,
                )
                add_gaussian_noise_inplace(
                    clipped_sums, clip_norm=dp_clip_norm,
                    noise_multiplier=dp_noise_multiplier,
                )
                for param, grad_sum in zip(backbone_params, clipped_sums):
                    param.grad = (grad_sum / float(batch_size)).to(
                        dtype=param.dtype, device=param.device,
                    )
                optimizer.step()

            running_loss += loss.item() * batch_size
            running_samples += batch_size

        epoch_loss = running_loss
        epoch_samples = running_samples

    avg_loss = epoch_loss / max(epoch_samples, 1)
    return {"loss": avg_loss, "num_samples": epoch_samples}


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
    dp_clip_norm: float = 1.0,
    dp_noise_multiplier: float = 0.0,
    device: torch.device = torch.device("cpu"),
    init_dataloader: DataLoader | None = None,
    embedding_init_log_every: int = 0,
    use_amp: bool = False,
    grad_scaler: Optional["torch.cuda.amp.GradScaler"] = None,
) -> dict:
    if round_num == 0:
        initialize_client_embedding(
            model,
            init_dataloader or dataloader,
            client_id,
            device,
            log_every_batches=embedding_init_log_every,
            log_prefix="  ",
        )

    return train_one_round(
        model=model,
        dataloader=dataloader,
        client_id=client_id,
        local_epochs=local_epochs,
        lr=lr,
        margin=margin,
        dp_clip_norm=dp_clip_norm,
        dp_noise_multiplier=dp_noise_multiplier,
        device=device,
        use_amp=use_amp,
        grad_scaler=grad_scaler,
    )
