"""Evaluate a saved simulation checkpoint on unseen images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from federated_project.dataset import FaceDataset, get_default_transforms
from federated_project.federation import create_model, resolve_device


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on unseen images.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional path to save evaluation report JSON.",
    )
    return parser


def count_images(directory: Path) -> int:
    return sum(
        1
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    eval_dir: str,
    batch_size: int,
    num_workers: int,
    device: str | None,
) -> dict[str, object]:
    resolved_device = resolve_device(device)
    ckpt = torch.load(checkpoint_path, map_location=resolved_device)

    class_names = list(ckpt["class_names"])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    model = create_model(
        num_clients=int(ckpt["num_clients"]),
        pretrained=str(ckpt["pretrained"]),
        device=resolved_device,
    )
    model.feature_extractor.load_state_dict(ckpt["feature_extractor_state_dict"], strict=True)

    saved_W = ckpt["W_matrix"].to(device=resolved_device, dtype=model.W_matrix.dtype)
    model.W_matrix.data.copy_(F.normalize(saved_W, p=2, dim=1))
    model.eval()

    eval_root = Path(eval_dir)
    eval_classes = sorted(
        entry.name
        for entry in eval_root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )

    seen_classes = [name for name in eval_classes if name in class_to_idx]
    missing_classes = [name for name in eval_classes if name not in class_to_idx]

    total_samples = 0
    total_correct = 0
    per_class: list[dict[str, object]] = []

    normalized_W = F.normalize(model.W_matrix, p=2, dim=1)

    for class_name in seen_classes:
        class_dir = eval_root / class_name
        true_id = class_to_idx[class_name]

        dataset = FaceDataset(
            root_dir=str(class_dir),
            client_id=true_id,
            transform=get_default_transforms(train=False),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        class_total = 0
        class_correct = 0
        class_true_similarity_sum = 0.0

        for images, _ in loader:
            images = images.to(resolved_device)
            features = model(images)
            similarities = features @ normalized_W.T
            predictions = similarities.argmax(dim=1)

            class_total += int(images.size(0))
            class_correct += int((predictions == true_id).sum().item())
            class_true_similarity_sum += float(similarities[:, true_id].sum().item())

        class_accuracy = class_correct / max(class_total, 1)
        class_avg_similarity = class_true_similarity_sum / max(class_total, 1)

        per_class.append(
            {
                "class_name": class_name,
                "num_images": class_total,
                "accuracy": class_accuracy,
                "avg_true_similarity": class_avg_similarity,
            }
        )

        total_samples += class_total
        total_correct += class_correct

    overall_accuracy = total_correct / max(total_samples, 1)

    return {
        "checkpoint": str(checkpoint_path),
        "eval_dir": str(eval_dir),
        "num_known_eval_classes": len(seen_classes),
        "num_unknown_eval_classes": len(missing_classes),
        "unknown_eval_classes": missing_classes,
        "num_samples": total_samples,
        "overall_top1_accuracy": overall_accuracy,
        "per_class": per_class,
    }


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate(
        checkpoint_path=args.checkpoint,
        eval_dir=args.eval_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    print(json.dumps(report, indent=2))

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
