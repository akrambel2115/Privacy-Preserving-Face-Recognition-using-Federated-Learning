"""CLI entrypoint for running face recognition with a trained model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from federated_project.dataset import get_default_transforms
from federated_project.federation import create_model, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict the person shown in a face image using the trained federated model."
    )
    parser.add_argument("image_path", help="Path to the face image to classify.")
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data" / "students"),
        help=(
            "Root directory used during training, with one subdirectory per enrolled person. "
            "The sorted subdirectory names are used to map class IDs back to person names."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a saved .pt model file. Defaults to the latest file in models/.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Unknown-person threshold on cosine similarity. Scores below this are labeled Unknown.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use, for example cpu or cuda. Defaults to cuda when available.",
    )
    return parser


def get_person_names(data_dir: str | Path) -> list[str]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory '{root}' does not exist.")

    person_names = sorted(
        directory.name
        for directory in root.iterdir()
        if directory.is_dir() and not directory.name.startswith(".")
    )
    if not person_names:
        raise FileNotFoundError(
            f"No person subdirectories found under '{root}'. "
            "Expected one subdirectory per enrolled person."
        )

    return person_names


def resolve_model_path(model_path: str | None) -> Path:
    if model_path is not None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file '{path}' does not exist.")
        if not path.is_file():
            raise FileNotFoundError(f"Model path '{path}' is not a file.")
        return path

    candidates = sorted(MODELS_DIR.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No saved model files were found in '{MODELS_DIR}'. "
            "Run training first or pass --model-path explicitly."
        )

    return candidates[-1]


def load_trained_model(
    model_path: Path,
    num_clients: int,
    device: torch.device,
) -> torch.nn.Module:
    # The saved checkpoint already contains all learned weights, so we can
    # rebuild the architecture without downloading pretrained initialization.
    model = create_model(
        num_clients=num_clients,
        pretrained=None,
        device=device,
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_image(
    model: torch.nn.Module,
    image_path: str | Path,
    device: torch.device,
) -> tuple[int, float]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file '{path}' does not exist.")
    if not path.is_file():
        raise FileNotFoundError(f"Image path '{path}' is not a file.")

    transform = get_default_transforms(train=False)
    image = Image.open(path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(image_tensor).squeeze(0)
        class_embeddings = F.normalize(model.W_matrix, p=2, dim=1)
        scores = torch.matmul(class_embeddings, embedding)
        best_score, best_index = torch.max(scores, dim=0)

    return int(best_index.item()), float(best_score.item())


def main() -> None:
    args = build_parser().parse_args()

    person_names = get_person_names(args.data_dir)
    model_path = resolve_model_path(args.model_path)
    device = resolve_device(args.device)

    model = load_trained_model(
        model_path=model_path,
        num_clients=len(person_names),
        device=device,
    )
    best_index, similarity = predict_image(
        model=model,
        image_path=args.image_path,
        device=device,
    )

    predicted_name = person_names[best_index] if similarity >= args.threshold else "Unknown"

    print(f"Image: {Path(args.image_path)}")
    print(f"Model: {model_path}")
    print(f"Predicted person: {predicted_name}")
    print(f"Similarity score: {similarity:.6f}")
    print(f"Threshold: {args.threshold:.6f}")

    if predicted_name == "Unknown":
        print(f"Best known match: {person_names[best_index]}")


if __name__ == "__main__":
    main()
