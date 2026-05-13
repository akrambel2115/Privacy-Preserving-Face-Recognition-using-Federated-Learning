"""CLI tool for 1:1 face verification.

Given two face images and a trained checkpoint, detect and crop faces using
MTCNN, extract embeddings, and report whether they depict the same person
based on cosine similarity.

Example:
  .\\.venv312\\Scripts\\python.exe scripts\\verify_pair.py ^
      custom_test\\1.jpg custom_test\\2.jpg ^
      --checkpoint custom_test\\best_run.pt ^
      --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from facenet_pytorch import MTCNN
from federated_project.dataset import get_default_transforms
from federated_project.federation import create_model, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify whether two face images belong to the same person.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/verify_pair.py img1.jpg img2.jpg --checkpoint model.pt\n"
            "  python scripts/verify_pair.py img1.jpg img2.jpg --checkpoint model.pt --threshold 0.5\n"
        ),
    )
    parser.add_argument("image_a", help="Path to the first face image.")
    parser.add_argument("image_b", help="Path to the second face image.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a benchmark-suite checkpoint (.pt) file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help=(
            "Cosine similarity threshold for same-person decision. "
            "Scores >= threshold -> SAME PERSON. (default: 0.6)"
        ),
    )
    parser.add_argument(
        "--no-face-detect",
        action="store_true",
        help="Skip MTCNN face detection and just resize the whole image (not recommended).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cpu or cuda. (default: auto-detect)",
    )
    return parser


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a model from a benchmark-suite checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = create_model(
        num_clients=int(ckpt["num_clients"]),
        pretrained=str(ckpt["pretrained"]),
        device=device,
    )
    model.feature_extractor.load_state_dict(
        ckpt["feature_extractor_state_dict"], strict=True
    )

    saved_W = ckpt["W_matrix"].to(device=device, dtype=model.W_matrix.dtype)
    model.W_matrix.data.copy_(F.normalize(saved_W, p=2, dim=1))
    model.eval()
    return model


def detect_and_crop_face(
    image_path: Path,
    mtcnn: MTCNN,
    device: torch.device,
) -> torch.Tensor | None:
    """Detect a face using MTCNN and return a cropped+aligned 160x160 tensor.

    Returns None if no face is detected.
    """
    img = Image.open(image_path).convert("RGB")
    # MTCNN returns a tensor of shape (3, 160, 160) with values in [-1, 1]
    # when image_size=160 and post_process=True (the default normalization).
    face_tensor = mtcnn(img)
    return face_tensor


@torch.no_grad()
def extract_embedding(
    model: torch.nn.Module,
    face_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Extract a unit-normalized 512-D embedding from a preprocessed face tensor."""
    tensor = face_tensor.unsqueeze(0).to(device)
    embedding = model(tensor).squeeze(0)  # already L2-normalized by the model
    return embedding


@torch.no_grad()
def extract_embedding_no_detect(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    transform,
) -> torch.Tensor:
    """Extract embedding without face detection (just resize whole image)."""
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    embedding = model(tensor).squeeze(0)
    return embedding


def main() -> None:
    args = build_parser().parse_args()

    # Validate paths
    path_a = Path(args.image_a)
    path_b = Path(args.image_b)
    ckpt_path = Path(args.checkpoint)

    for path, label in [(path_a, "Image A"), (path_b, "Image B"), (ckpt_path, "Checkpoint")]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    device = resolve_device(args.device)
    model = load_checkpoint_model(ckpt_path, device)

    if args.no_face_detect:
        # Legacy mode: just resize the whole image
        transform = get_default_transforms(train=False)
        emb_a = extract_embedding_no_detect(model, path_a, device, transform)
        emb_b = extract_embedding_no_detect(model, path_b, device, transform)
        detect_mode = "disabled (whole-image resize)"
    else:
        # MTCNN face detection + alignment
        mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,  # normalizes to [-1, 1]
            device=device,
        )

        face_a = detect_and_crop_face(path_a, mtcnn, device)
        face_b = detect_and_crop_face(path_b, mtcnn, device)

        if face_a is None:
            print(f"ERROR: No face detected in Image A: {path_a}")
            print("Try --no-face-detect to skip face detection.")
            sys.exit(1)
        if face_b is None:
            print(f"ERROR: No face detected in Image B: {path_b}")
            print("Try --no-face-detect to skip face detection.")
            sys.exit(1)

        emb_a = extract_embedding(model, face_a, device)
        emb_b = extract_embedding(model, face_b, device)
        detect_mode = "MTCNN (aligned crop)"

    # Cosine similarity (embeddings are already unit-normed)
    similarity = float(torch.dot(emb_a, emb_b).item())
    is_same = similarity >= args.threshold

    # Pretty output
    verdict = "[YES] SAME PERSON" if is_same else "[NO]  DIFFERENT PERSON"
    bar_length = 40
    filled = int(max(0.0, min(1.0, (similarity + 1) / 2)) * bar_length)  # map [-1,1] -> [0,1]
    bar = "#" * filled + "-" * (bar_length - filled)

    print()
    print("=" * 60)
    print("  Face Verification Result")
    print("=" * 60)
    print(f"  Image A    : {path_a}")
    print(f"  Image B    : {path_b}")
    print(f"  Model      : {ckpt_path.name}")
    print(f"  Face Detect: {detect_mode}")
    print("-" * 60)
    print(f"  Similarity : {similarity:+.6f}")
    print(f"  Threshold  : {args.threshold:+.6f}")
    print(f"  Confidence : [{bar}]")
    print("-" * 60)
    print(f"  Verdict    : {verdict}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
