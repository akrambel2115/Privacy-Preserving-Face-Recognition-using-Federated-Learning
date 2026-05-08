"""Fused within-round client training.

This module provides an alternative to the sequential per-client loop in
``train.py``. Instead of training 1000 clients one at a time within a
round, it stacks one batch from every active client into a single
mega-batch and runs ONE forward+backward pass through the backbone per
step. The per-client class embeddings (rows of W) are looked up by index
and the squared-hinge loss is computed per-row, then averaged.

Why this works without changing the algorithm fundamentally:
  * In the FedFace algorithm, every client in round t starts from the
    SAME global state (theta_t, W_t). Sequential per-client training
    diverges from this only because each client takes multiple local
    SGD steps before the others. With ``local_epochs=1`` and the same
    optimizer, sequential and fused training compute the exact same
    gradient SUM at each step -- they just compute it client-by-client
    in sequential, and all-at-once in fused.
  * For ``local_epochs > 1`` the two paths diverge slightly: sequential
    lets each client take its own k local steps before the others; fused
    interleaves them (epoch 1 across all clients, then epoch 2, etc.).
    For 1 local epoch they're equivalent. For more, fused is roughly
    equivalent to FedAvg with k synchronous mini-rounds inside a logical
    "round" -- a defensible variant but not the literal paper procedure.
  * We use SGD (no momentum by default) so the optimizer is stateless
    across logical "clients within a step" -- this matches the math of
    summing per-client gradients exactly. Adam is *not* used here because
    its per-parameter state would entangle clients that share the
    optimizer; that would be a real algorithmic deviation.

Limitations / gates:
  * Disabled when DP is on (per-sample gradient clipping is incompatible
    with the fused loss formulation).
  * Spreadout regularization runs at the end of the round on the W matrix
    (same as sequential).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from federated_project.dataset import (
    FaceDataset,
    get_default_transforms,
    list_client_dirs,
)


# ---------------------------------------------------------------------------
# Combined dataset that emits (image, client_id) for many clients at once
# ---------------------------------------------------------------------------

class MultiClientDataset(Dataset):
    """Concatenated dataset over a SUBSET of clients for one round.

    Each item is (image_tensor, client_id_int). When fed through a normal
    DataLoader with shuffle=True, batches will naturally contain mixed
    clients -- which is exactly what the fused training loop wants.
    """

    def __init__(self, client_dirs: list, client_ids: list[int], transform):
        if len(client_dirs) != len(client_ids):
            raise ValueError("client_dirs and client_ids must have the same length")
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        for cdir, cid in zip(client_dirs, client_ids):
            sub = FaceDataset(root_dir=str(cdir), client_id=cid, transform=transform)
            for path, _ in sub.samples:
                self.samples.append((str(path), int(cid)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, cid = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, cid


def build_round_loader(
    data_dir: str,
    active_client_ids: list[int],
    train: bool,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Build one DataLoader over all images of all active clients in this round.

    The yielded batches contain a mix of client_ids; the fused training
    loop dispatches on client_id within the batch.
    """
    all_dirs = list_client_dirs(data_dir)
    if not all_dirs:
        raise FileNotFoundError(
            f"No client subdirectories under '{data_dir}'. Expected multi-person layout."
        )
    selected_dirs = [all_dirs[cid] for cid in active_client_ids]
    transform = get_default_transforms(train=train)
    ds = MultiClientDataset(selected_dirs, active_client_ids, transform)

    kwargs = {
        "batch_size": batch_size,
        "shuffle": train,
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,  # keep all samples in fused mode
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = False  # round loader is short-lived
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


# ---------------------------------------------------------------------------
# Fused round
# ---------------------------------------------------------------------------

@dataclass
class FusedRoundResult:
    train_loss: float       # mean per-sample squared-hinge loss across the round
    num_samples: int        # total samples processed
    n_steps: int            # total optimizer steps taken


def _fused_loss(
    features: torch.Tensor,         # (B, d), L2-normalized
    W: torch.Tensor,                # (C, d), normalized per-row outside
    client_ids: torch.Tensor,       # (B,) long
    margin: float,
) -> torch.Tensor:
    """Per-sample squared hinge loss against each sample's own anchor.

    For each sample j, looks up w_{c_j} = W[client_ids[j]] and computes
    max(0, m - w_{c_j} . f_j)^2. Returns the mean over the batch.
    """
    # Look up per-sample anchors: (B, d)
    anchors = F.normalize(W[client_ids], p=2, dim=1)
    # Cosine = elementwise product summed over d
    cos_sim = (features * anchors).sum(dim=1)
    hinge = torch.clamp(margin - cos_sim, min=0.0)
    return (hinge ** 2).mean()


def train_round_fused(
    model: nn.Module,
    data_dir: str,
    active_client_ids: list[int],
    local_epochs: int,
    lr: float,
    margin: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool = False,
    grad_scaler: Optional["torch.cuda.amp.GradScaler"] = None,
    train_augment: bool = True,
) -> FusedRoundResult:
    """Run one fused federated round.

    All ``active_client_ids`` train SIMULTANEOUSLY against their own anchor
    rows of ``model.W_matrix``. The backbone receives the SUM of per-client
    gradients each step (via the per-sample squared-hinge loss aggregated
    over a mixed-client batch).

    Algorithmic notes:
      - Optimizer: plain SGD on (backbone params + W_matrix). No momentum,
        no Adam state -- so summing per-client gradients in a mega-batch
        is mathematically equivalent to summing per-client SGD updates.
      - With local_epochs > 1: we iterate the full round-loader k times.
        Each pass is one synchronous mini-step across all clients.
    """
    model.train()
    if device.type == "cuda":
        amp_active = bool(use_amp)
    else:
        amp_active = False
    if amp_active and grad_scaler is None:
        grad_scaler = torch.cuda.amp.GradScaler()

    # Trainable params: unfrozen backbone + W_matrix.
    # SGD: stateless w.r.t. parameter identity -> per-client gradient sum is
    # exactly the per-client update sum, by linearity. (Linearity holds for
    # SGD without momentum; would NOT hold for Adam, hence the choice.)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=lr)

    loader = build_round_loader(
        data_dir=data_dir,
        active_client_ids=active_client_ids,
        train=train_augment,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    total_loss = 0.0
    total_samples = 0
    n_steps = 0

    for _epoch in range(int(local_epochs)):
        for images, client_ids in loader:
            images = images.to(device, non_blocking=True)
            client_ids = client_ids.to(device, non_blocking=True).long()
            B = images.size(0)

            optimizer.zero_grad(set_to_none=True)

            if amp_active:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = model(images)
                features = features.float()  # loss in fp32 for stability
                loss = _fused_loss(features, model.W_matrix, client_ids, margin)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                features = model(images)
                loss = _fused_loss(features, model.W_matrix, client_ids, margin)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * B
            total_samples += B
            n_steps += 1

    avg_loss = total_loss / max(total_samples, 1)
    return FusedRoundResult(
        train_loss=avg_loss,
        num_samples=total_samples,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Mean Feature Initialization across many clients in one pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def fused_initialize_embeddings(
    model: nn.Module,
    data_dir: str,
    client_ids: list[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    log: bool = True,
) -> None:
    """Compute Mean Feature Initialization (paper Eq. 6) for many clients
    in one streaming pass over the un-augmented combined loader.

    Equivalent to looping ``initialize_client_embedding`` per client, but
    using a single shared backbone evaluation per image.
    """
    model.eval()
    loader = build_round_loader(
        data_dir=data_dir,
        active_client_ids=client_ids,
        train=False,            # un-augmented
        batch_size=batch_size,
        num_workers=num_workers,
    )

    d = model.W_matrix.size(1)
    sums = torch.zeros(model.W_matrix.size(0), d, device=device, dtype=torch.float32)
    counts = torch.zeros(model.W_matrix.size(0), device=device, dtype=torch.float32)

    t0 = time.perf_counter()
    n_total = 0
    for images, cids in loader:
        images = images.to(device, non_blocking=True)
        cids = cids.to(device, non_blocking=True).long()
        feats = model(images).float()  # already L2-normalized
        sums.index_add_(0, cids, feats)
        counts.index_add_(0, cids, torch.ones_like(cids, dtype=torch.float32))
        n_total += int(images.size(0))

    valid = counts > 0
    means = torch.zeros_like(sums)
    means[valid] = sums[valid] / counts[valid].unsqueeze(1)
    means = F.normalize(means, p=2, dim=1)

    # Only overwrite W rows for clients that actually had samples in this pass.
    for cid in client_ids:
        if counts[cid].item() > 0:
            model.W_matrix.data[cid] = means[cid]

    if log:
        elapsed = time.perf_counter() - t0
        print(
            f"  fused MFI: {len(client_ids)} clients, {n_total} images "
            f"in {elapsed:.1f}s"
        )