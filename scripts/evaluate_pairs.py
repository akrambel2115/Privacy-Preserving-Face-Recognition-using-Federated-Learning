"""Evaluate an embedding model with genuine/impostor cosine pairs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from federated_project.dataset import get_default_transforms
from federated_project.federation import create_model, resolve_device


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


@dataclass(frozen=True)
class EmbeddedImage:
    class_name: str
    path: str
    embedding: torch.Tensor


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[Path]) -> None:
        self.image_paths = image_paths
        self.transform = get_default_transforms(train=False)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate face embeddings using same-person and different-person "
            "cosine-similarity pairs."
        )
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional checkpoint or full model .pt file. Omit to evaluate pretrained FaceNet.",
    )
    parser.add_argument("--pretrained", default="vggface2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--max-pairs-per-kind", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", default=None)
    return parser


def collect_image_paths(
    data_dir: str | Path,
    max_images_per_class: int | None,
    seed: int,
) -> dict[str, list[Path]]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory '{root}' does not exist.")

    rng = random.Random(seed)
    class_to_paths: dict[str, list[Path]] = {}
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if class_dir.name.startswith("."):
            continue
        paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
        if max_images_per_class is not None and len(paths) > max_images_per_class:
            paths = sorted(rng.sample(paths, max_images_per_class))
        if paths:
            class_to_paths[class_dir.name] = paths

    if len(class_to_paths) < 2:
        raise ValueError("Pair evaluation needs at least two classes with images.")

    return class_to_paths


def infer_num_clients_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int | None:
    w_matrix = state_dict.get("W_matrix")
    if isinstance(w_matrix, torch.Tensor) and w_matrix.ndim == 2:
        return int(w_matrix.shape[0])
    return None


def load_embedding_model(
    model_path: str | None,
    num_clients: int,
    pretrained: str,
    device: torch.device,
) -> torch.nn.Module:
    if model_path is None:
        model = create_model(
            num_clients=max(num_clients, 1),
            pretrained=pretrained,
            device=device,
            train_backbone=False,
        )
        model.eval()
        return model

    payload = torch.load(model_path, map_location=device)
    if isinstance(payload, dict) and "feature_extractor_state_dict" in payload:
        checkpoint_num_clients = int(payload.get("num_clients", num_clients))
        model = create_model(
            num_clients=checkpoint_num_clients,
            pretrained=None,
            device=device,
            train_backbone=False,
        )
        model.feature_extractor.load_state_dict(
            payload["feature_extractor_state_dict"],
            strict=True,
        )
        if "W_matrix" in payload:
            saved_w = payload["W_matrix"].to(device=device, dtype=model.W_matrix.dtype)
            model.W_matrix.data.copy_(F.normalize(saved_w, p=2, dim=1))
        model.eval()
        return model

    if not isinstance(payload, dict):
        raise ValueError("Expected a checkpoint dictionary or model state_dict.")

    checkpoint_num_clients = infer_num_clients_from_state_dict(payload) or num_clients
    model = create_model(
        num_clients=checkpoint_num_clients,
        pretrained=None,
        device=device,
        train_backbone=False,
    )
    model.load_state_dict(payload, strict=True)
    model.eval()
    return model


@torch.no_grad()
def embed_dataset(
    model: torch.nn.Module,
    class_to_paths: dict[str, list[Path]],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[EmbeddedImage]:
    flat_items = [
        (class_name, path)
        for class_name, paths in class_to_paths.items()
        for path in paths
    ]
    dataset = ImagePathDataset([path for _class_name, path in flat_items])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    embeddings: list[torch.Tensor] = []
    for images in loader:
        images = images.to(device)
        batch_embeddings = model(images).cpu()
        embeddings.extend(batch_embeddings)

    return [
        EmbeddedImage(
            class_name=class_name,
            path=str(path),
            embedding=embedding,
        )
        for (class_name, path), embedding in zip(flat_items, embeddings)
    ]


def sample_pair_scores(
    embedded_images: list[EmbeddedImage],
    max_pairs_per_kind: int,
    seed: int,
) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)
    by_class: dict[str, list[EmbeddedImage]] = {}
    for item in embedded_images:
        by_class.setdefault(item.class_name, []).append(item)

    same_pairs: list[tuple[EmbeddedImage, EmbeddedImage]] = []
    for items in by_class.values():
        same_pairs.extend(combinations(items, 2))

    class_names = sorted(by_class)
    diff_possible = sum(
        len(by_class[left]) * len(by_class[right])
        for left_idx, left in enumerate(class_names)
        for right in class_names[left_idx + 1 :]
    )

    if len(same_pairs) > max_pairs_per_kind:
        same_pairs = rng.sample(same_pairs, max_pairs_per_kind)

    if diff_possible <= max_pairs_per_kind:
        diff_pairs = [
            (left_item, right_item)
            for left_idx, left_name in enumerate(class_names)
            for right_name in class_names[left_idx + 1 :]
            for left_item in by_class[left_name]
            for right_item in by_class[right_name]
        ]
    else:
        diff_pairs = []
        seen: set[tuple[str, str]] = set()
        while len(diff_pairs) < max_pairs_per_kind:
            left_name, right_name = rng.sample(class_names, 2)
            left_item = rng.choice(by_class[left_name])
            right_item = rng.choice(by_class[right_name])
            key = tuple(sorted((left_item.path, right_item.path)))
            if key in seen:
                continue
            seen.add(key)
            diff_pairs.append((left_item, right_item))

    return _cosine_scores(same_pairs), _cosine_scores(diff_pairs)


def _cosine_scores(pairs: list[tuple[EmbeddedImage, EmbeddedImage]]) -> list[float]:
    return [
        float(torch.dot(left.embedding, right.embedding).item())
        for left, right in pairs
    ]


def summarize_scores(scores: list[float]) -> dict[str, float | int | None]:
    if not scores:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    values = np.asarray(scores, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def threshold_metrics(
    same_scores: list[float],
    diff_scores: list[float],
    threshold: float,
) -> dict[str, float]:
    positives = np.asarray(same_scores, dtype=np.float64)
    negatives = np.asarray(diff_scores, dtype=np.float64)
    true_accepts = int((positives >= threshold).sum())
    false_rejects = int((positives < threshold).sum())
    false_accepts = int((negatives >= threshold).sum())
    true_rejects = int((negatives < threshold).sum())
    total = len(positives) + len(negatives)
    return {
        "threshold": float(threshold),
        "accuracy": (true_accepts + true_rejects) / max(total, 1),
        "true_accept_rate": true_accepts / max(len(positives), 1),
        "false_reject_rate": false_rejects / max(len(positives), 1),
        "false_accept_rate": false_accepts / max(len(negatives), 1),
        "true_reject_rate": true_rejects / max(len(negatives), 1),
    }


def best_threshold_metrics(
    same_scores: list[float],
    diff_scores: list[float],
) -> dict[str, float]:
    thresholds = np.linspace(-1.0, 1.0, num=1001)
    best = threshold_metrics(same_scores, diff_scores, float(thresholds[0]))
    for threshold in thresholds[1:]:
        current = threshold_metrics(same_scores, diff_scores, float(threshold))
        if current["accuracy"] > best["accuracy"]:
            best = current
    return best


def roc_auc(same_scores: list[float], diff_scores: list[float]) -> float | None:
    if not same_scores or not diff_scores:
        return None

    scores = np.asarray(same_scores + diff_scores, dtype=np.float64)
    labels = np.asarray([1] * len(same_scores) + [0] * len(diff_scores), dtype=np.int64)
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=np.float64)

    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_ranks = ranks[labels == 1].sum()
    num_positive = len(same_scores)
    num_negative = len(diff_scores)
    auc = (
        positive_ranks - num_positive * (num_positive + 1) / 2.0
    ) / (num_positive * num_negative)
    return float(auc)


def evaluate_pairs(args: argparse.Namespace) -> dict[str, object]:
    device = resolve_device(args.device)
    class_to_paths = collect_image_paths(
        data_dir=args.data_dir,
        max_images_per_class=args.max_images_per_class,
        seed=args.seed,
    )
    model = load_embedding_model(
        model_path=args.model_path,
        num_clients=len(class_to_paths),
        pretrained=args.pretrained,
        device=device,
    )
    embedded_images = embed_dataset(
        model=model,
        class_to_paths=class_to_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    same_scores, diff_scores = sample_pair_scores(
        embedded_images=embedded_images,
        max_pairs_per_kind=args.max_pairs_per_kind,
        seed=args.seed,
    )

    return {
        "data_dir": str(args.data_dir),
        "model_path": str(args.model_path) if args.model_path else None,
        "num_classes": len(class_to_paths),
        "num_images": len(embedded_images),
        "same_person": summarize_scores(same_scores),
        "different_person": summarize_scores(diff_scores),
        "threshold_metrics": threshold_metrics(
            same_scores=same_scores,
            diff_scores=diff_scores,
            threshold=args.threshold,
        ),
        "best_threshold_metrics": best_threshold_metrics(
            same_scores=same_scores,
            diff_scores=diff_scores,
        ),
        "roc_auc": roc_auc(same_scores, diff_scores),
    }


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_pairs(args)
    print(json.dumps(report, indent=2))

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
