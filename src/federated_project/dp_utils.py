"""Differential privacy utilities (math only, no Flower logic).

This module centralizes all DP-related computations for the FedFace project:
- DP-SGD gradient clipping + Gaussian noise injection (backbone parameters only)
- Anchor (w_i) Gaussian mechanism
- Lightweight (ε, δ) accounting via an RDP accountant (Opacus, if installed)
- TAR@FAR evaluation helpers for face verification
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Anchor privatization (Gaussian mechanism)
# ---------------------------------------------------------------------------


def privatize_anchor(
    anchor: np.ndarray,
    n_local_samples: int,
    noise_multiplier: float,
    sensitivity_override: float | None = None,
) -> np.ndarray:
    """Return a noised (non-renormalized) version of a 512D anchor vector.

    Mechanism:
        w_private = w + η,   η ~ N(0, (σ * Δ)^2 I)

    Where:
        σ = noise_multiplier
        Δ = 2 / N  (global L2 sensitivity upper bound on unit-sphere outputs)

    Notes:
    - If noise_multiplier == 0.0, returns anchor unchanged.
    - Intentionally does NOT renormalize.
    """
    anchor = np.asarray(anchor, dtype=np.float32)
    if anchor.ndim != 1:
        raise ValueError(f"anchor must be 1D, got shape {anchor.shape}")

    if noise_multiplier == 0.0:
        return anchor

    if n_local_samples <= 0:
        raise ValueError("n_local_samples must be positive")

    sensitivity = (
        float(sensitivity_override)
        if sensitivity_override is not None
        else 2.0 / float(n_local_samples)
    )
    std_dev = float(noise_multiplier) * sensitivity

    eta = np.random.normal(loc=0.0, scale=std_dev, size=anchor.shape).astype(np.float32)
    noised = anchor + eta

    # Sanity check: if someone renormalized downstream, norm will return ~1.
    # This is intentionally a *weak* assertion (exact equality is enough to catch
    # accidental hard renormalization, while avoiding false positives for tiny noise).
    assert float(np.linalg.norm(noised)) != 1.0

    return noised


# ---------------------------------------------------------------------------
# DP-SGD backbone gradient processing (per-sample clipping)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DpSgdConfig:
    clip_norm: float
    noise_multiplier: float


def _global_l2_norm(grads: Sequence[torch.Tensor | None]) -> torch.Tensor:
    squared = None
    for grad in grads:
        if grad is None:
            continue
        value = (grad.detach() ** 2).sum()
        squared = value if squared is None else squared + value
    if squared is None:
        return torch.tensor(0.0)
    return torch.sqrt(squared + 1e-12)


def compute_clipped_grad_sum(
    per_sample_losses: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    clip_norm: float,
) -> list[torch.Tensor]:
    """Compute sum of per-sample clipped gradients for a batch.

    Returns a list of tensors aligned with `params` containing:
        Σ_i clip_C(∇_θ loss_i)

    This is the DP-SGD core primitive; caller may add Gaussian noise and
    divide by batch size to obtain the final gradient estimate.
    """
    if per_sample_losses.ndim != 1:
        raise ValueError(
            f"per_sample_losses must have shape (B,), got {tuple(per_sample_losses.shape)}"
        )
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")

    grad_sums = [torch.zeros_like(param) for param in params]

    # Looping per-sample is slower than vmap, but is explicit and avoids
    # introducing additional framework dependencies.
    for loss_i in per_sample_losses:
        grads_i = torch.autograd.grad(
            loss_i,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        norm = _global_l2_norm(grads_i)
        if float(norm.item()) == 0.0:
            continue
        scale = min(1.0, float(clip_norm) / float(norm.item()))
        for idx, grad in enumerate(grads_i):
            if grad is None:
                continue
            grad_sums[idx].add_(grad.detach(), alpha=scale)

    return grad_sums


def add_gaussian_noise_inplace(
    grad_sums: Sequence[torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
    generator: torch.Generator | None = None,
) -> None:
    """Add isotropic Gaussian noise to each tensor in `grad_sums` in-place."""
    if noise_multiplier == 0.0:
        return
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")

    std = float(noise_multiplier) * float(clip_norm)
    for tensor in grad_sums:
        # torch.randn_like does not support `generator=` on some torch builds.
        if generator is None:
            noise = torch.randn_like(tensor)
        else:
            noise = torch.randn(
                tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
                generator=generator,
            )
        tensor.add_(noise * std)


# ---------------------------------------------------------------------------
# Privacy accounting (RDP)
# ---------------------------------------------------------------------------


def compute_epsilon(
    num_rounds: int,
    noise_multiplier: float,
    clip_norm: float,
    dataset_size: int,
    batch_size: int,
    delta: float = 1e-5,
) -> float:
    """Compute ε for a DP-SGD-style mechanism using an RDP accountant.

    This uses Opacus' standalone accountant if available.

    Important: this is an *accounting helper* with a simplified interface.
    It assumes one full pass worth of SGD steps per round:
        steps_per_round ≈ ceil(dataset_size / batch_size)

    If your training uses multiple local epochs per round, account for that
    externally by increasing `num_rounds` accordingly.
    """
    if noise_multiplier == 0.0:
        return float("inf")
    if num_rounds <= 0:
        raise ValueError("num_rounds must be positive")
    if dataset_size <= 0 or batch_size <= 0:
        raise ValueError("dataset_size and batch_size must be positive")
    if batch_size > dataset_size:
        # Still valid; sampling rate capped at 1.0
        batch_size = dataset_size

    try:
        from opacus.accountants.rdp import RDPAccountant
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Opacus is required for compute_epsilon(). Install with: pip install opacus"
        ) from exc

    sampling_rate = float(batch_size) / float(dataset_size)
    steps_per_round = int(math.ceil(float(dataset_size) / float(batch_size)))
    total_steps = int(num_rounds) * steps_per_round

    accountant = RDPAccountant()
    # Opacus expects noise multiplier for Gaussian mechanism.
    for _ in range(total_steps):
        accountant.step(noise_multiplier=float(noise_multiplier), sample_rate=sampling_rate)

    epsilon = float(accountant.get_epsilon(delta=float(delta)))
    return epsilon


# ---------------------------------------------------------------------------
# TAR@FAR evaluation helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalPair:
    image_a: Path
    image_b: Path
    is_same_person: bool


def load_eval_pairs(pairs_file: str | Path, image_dir: str | Path) -> list[EvalPair]:
    """Load verification pairs.

    Supported formats:

    1) Simple labeled paths (CSV/TSV-ish):
        rel_path_a, rel_path_b, is_same(0/1)

    2) LFW pairs.txt-style (space-separated):
        first line: number of pairs
        subsequent lines are either:
          - same:  name idx1 idx2
          - diff:  name1 idx1 name2 idx2
        resolved as: name/name_####.jpg under image_dir

    3) Kaggle LFW pairs.csv (comma-separated, header, trailing commas possible):
        header: name,imagenum1,imagenum2,
        same:   name,idx1,idx2,
        diff:   name1,idx1,name2,idx2

    Paths are resolved relative to `image_dir`.
    """

    def _looks_like_path(value: str) -> bool:
        lowered = value.lower()
        return (
            "/" in value
            or "\\" in value
            or lowered.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        )

    def _try_int(value: str) -> int | None:
        try:
            return int(value)
        except Exception:
            return None

    pairs_path = Path(pairs_file)
    root = Path(image_dir)
    lines = [ln.strip() for ln in pairs_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return []

    # LFW pairs file often begins with a count
    start_idx = 1 if lines[0].isdigit() else 0

    pairs: list[EvalPair] = []

    for line in lines[start_idx:]:
        # Normalize to comma-separated, but preserve empty trailing field(s) so we can
        # correctly handle Kaggle CSV lines with a trailing comma.
        raw_parts = [p.strip() for p in line.replace("\t", ",").split(",")]
        while raw_parts and raw_parts[-1] == "":
            raw_parts.pop()

        if not raw_parts:
            continue

        # Skip Kaggle header row (and similar CSV headers)
        joined = ",".join(raw_parts).lower()
        if raw_parts[0].lower() in {"name", "name1"} or "imagenum" in joined:
            continue

        if len(raw_parts) == 3:
            first, second, third = raw_parts
            idx1 = _try_int(second)
            idx2_or_label = _try_int(third)
            if idx1 is not None and idx2_or_label is not None and not _looks_like_path(first):
                # Kaggle/LFW same-person row: name, idx1, idx2
                name = first
                pairs.append(
                    EvalPair(
                        image_a=_lfw_path(root, name, idx1),
                        image_b=_lfw_path(root, name, idx2_or_label),
                        is_same_person=True,
                    )
                )
            else:
                # Labeled relative paths: rel_a, rel_b, is_same(0/1)
                a, b, same = raw_parts
                pairs.append(
                    EvalPair(
                        image_a=(root / a),
                        image_b=(root / b),
                        is_same_person=bool(int(same)),
                    )
                )
            continue

        if len(raw_parts) == 4:
            p0, p1, p2, p3 = raw_parts
            idx1 = _try_int(p1)
            idx2 = _try_int(p3)

            # Kaggle/LFW different-person row: name1, idx1, name2, idx2
            if (
                idx1 is not None
                and idx2 is not None
                and not _looks_like_path(p0)
                and not _looks_like_path(p2)
            ):
                pairs.append(
                    EvalPair(
                        image_a=_lfw_path(root, p0, idx1),
                        image_b=_lfw_path(root, p2, idx2),
                        is_same_person=False,
                    )
                )
                continue

            # LFW-style same-person with explicit label: name, idx1, idx2, label
            idx2_alt = _try_int(p2)
            if idx1 is not None and idx2_alt is not None and p3 in {"0", "1"} and not _looks_like_path(p0):
                pairs.append(
                    EvalPair(
                        image_a=_lfw_path(root, p0, idx1),
                        image_b=_lfw_path(root, p0, idx2_alt),
                        is_same_person=bool(int(p3)),
                    )
                )
                continue

        if len(raw_parts) == 5:
            # LFW-style different-person line with label: name1, idx1, name2, idx2, label
            name1, idx1_s, name2, idx2_s, label = raw_parts
            idx1 = _try_int(idx1_s)
            idx2 = _try_int(idx2_s)
            if idx1 is None or idx2 is None:
                raise ValueError(f"Unrecognized pairs line: {line}")
            pairs.append(
                EvalPair(
                    image_a=_lfw_path(root, name1, idx1),
                    image_b=_lfw_path(root, name2, idx2),
                    is_same_person=bool(int(label)),
                )
            )
            continue

        # Raw LFW pairs format (space-separated) is common; fallback
        space_parts = line.split()
        if len(space_parts) == 3:
            name, idx1_s, idx2_s = space_parts
            pairs.append(
                EvalPair(
                    _lfw_path(root, name, int(idx1_s)),
                    _lfw_path(root, name, int(idx2_s)),
                    True,
                )
            )
        elif len(space_parts) == 4:
            name1, idx1_s, name2, idx2_s = space_parts
            pairs.append(
                EvalPair(
                    _lfw_path(root, name1, int(idx1_s)),
                    _lfw_path(root, name2, int(idx2_s)),
                    False,
                )
            )
        else:
            raise ValueError(f"Unrecognized pairs line: {line}")

    return pairs


def _lfw_path(root: Path, name: str, index: int) -> Path:
    return root / name / f"{name}_{index:04d}.jpg"


@torch.no_grad()
def compute_tar_at_far(
    model: torch.nn.Module,
    pairs: Sequence[EvalPair],
    far_target: float = 0.001,
    device: torch.device | None = None,
    transform: callable | None = None,
) -> float:
    """Compute TAR at a target FAR for face verification via cosine similarity."""
    if not pairs:
        raise ValueError("pairs must be non-empty")

    resolved_device = device or next(model.parameters()).device
    model.eval()

    # Lazy import to keep dp_utils standalone from torchvision unless used.
    from PIL import Image

    def default_transform(img):
        return img

    transform_fn = transform or default_transform

    embedding_cache: dict[Path, torch.Tensor] = {}

    def embed(path: Path) -> torch.Tensor:
        if path in embedding_cache:
            return embedding_cache[path]
        image = Image.open(path).convert("RGB")
        tensor = transform_fn(image)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("transform must return a torch.Tensor")
        tensor = tensor.unsqueeze(0).to(resolved_device)
        vec = model(tensor).squeeze(0).detach().cpu()
        embedding_cache[path] = vec
        return vec

    scores: list[float] = []
    labels: list[bool] = []

    for pair in pairs:
        emb_a = embed(pair.image_a)
        emb_b = embed(pair.image_b)
        score = float(torch.dot(emb_a, emb_b).item())
        scores.append(score)
        labels.append(bool(pair.is_same_person))

    scores_np = np.asarray(scores, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.bool_)

    genuine = scores_np[labels_np]
    impostor = scores_np[~labels_np]

    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("Pairs must include both genuine and impostor examples")

    # Candidate thresholds: sorted unique scores
    thresholds = np.unique(scores_np)
    thresholds.sort()

    far_values = []
    tar_values = []

    for t in thresholds:
        far = float(np.mean(impostor >= t))
        tar = float(np.mean(genuine >= t))
        far_values.append(far)
        tar_values.append(tar)

    far_arr = np.asarray(far_values, dtype=np.float64)
    tar_arr = np.asarray(tar_values, dtype=np.float64)

    # Find where FAR crosses the target
    below = np.where(far_arr <= far_target)[0]
    above = np.where(far_arr >= far_target)[0]

    if below.size == 0 and above.size == 0:
        return 0.0

    if above.size == 0:
        # FAR never reaches target (always below): return 0.0 per spec
        return 0.0

    if below.size == 0:
        # FAR always above target: return 1.0 per spec
        return 1.0

    lo = below[-1]
    hi = above[0]

    if lo == hi:
        return float(tar_arr[lo])

    far_lo, far_hi = float(far_arr[lo]), float(far_arr[hi])
    tar_lo, tar_hi = float(tar_arr[lo]), float(tar_arr[hi])

    if far_hi == far_lo:
        return float(tar_hi)

    # Linear interpolation
    alpha = (far_target - far_lo) / (far_hi - far_lo)
    return float(tar_lo + (tar_hi - tar_lo) * alpha)


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
