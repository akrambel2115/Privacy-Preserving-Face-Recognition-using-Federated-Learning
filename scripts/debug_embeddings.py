"""Diagnostic script: investigate why similarity is too high for different people."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from federated_project.dataset import get_default_transforms
from federated_project.model import FedFaceModel
from federated_project.federation import create_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGES = [
    PROJECT_ROOT / "custom_test" / "1.jpg",
    PROJECT_ROOT / "custom_test" / "2.jpg",
    PROJECT_ROOT / "custom_test" / "3.jpg",
    PROJECT_ROOT / "custom_test" / "4.jpg",
]

CHECKPOINT = PROJECT_ROOT / "custom_test" / "best_run.pt"


def load_finetuned_model():
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model = create_model(
        num_clients=int(ckpt["num_clients"]),
        pretrained=str(ckpt["pretrained"]),
        device=DEVICE,
    )
    model.feature_extractor.load_state_dict(ckpt["feature_extractor_state_dict"], strict=True)
    saved_W = ckpt["W_matrix"].to(device=DEVICE, dtype=model.W_matrix.dtype)
    model.W_matrix.data.copy_(F.normalize(saved_W, p=2, dim=1))
    model.eval()
    return model


def load_raw_pretrained_model():
    """Load raw VGGFace2 pretrained InceptionResnetV1 - NO fine-tuning."""
    model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
    return model


@torch.no_grad()
def embed_with_mtcnn(model, img_path, mtcnn):
    """MTCNN detect + crop, then embed."""
    img = Image.open(img_path).convert("RGB")
    face = mtcnn(img)
    if face is None:
        print(f"  WARNING: No face detected in {img_path.name}")
        return None
    tensor = face.unsqueeze(0).to(DEVICE)
    emb = model(tensor).squeeze(0)
    return F.normalize(emb, p=2, dim=0)


@torch.no_grad()
def embed_with_resize(model, img_path, transform):
    """Just resize whole image (training-style), then embed."""
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    emb = model(tensor).squeeze(0)
    return F.normalize(emb, p=2, dim=0)


def similarity_matrix(embeddings, labels):
    """Print a pairwise similarity matrix."""
    n = len(embeddings)
    print(f"  {'':>8}", end="")
    for j in range(n):
        print(f"  {labels[j]:>8}", end="")
    print()
    for i in range(n):
        print(f"  {labels[i]:>8}", end="")
        for j in range(n):
            sim = float(torch.dot(embeddings[i], embeddings[j]).item())
            print(f"  {sim:>8.4f}", end="")
        print()


def check_preprocessing_mismatch():
    """Compare MTCNN post_process normalization vs training Normalize."""
    print("\n" + "=" * 70)
    print("  TEST 1: Preprocessing Normalization Comparison")
    print("=" * 70)

    img = Image.open(IMAGES[0]).convert("RGB")
    img_np = np.array(img, dtype=np.float32)

    # MTCNN post_process: (img - 127.5) / 128.0
    mtcnn_norm = (img_np - 127.5) / 128.0

    # Training pipeline: ToTensor -> Normalize(0.5, 0.5)
    # ToTensor: img / 255.0  -> Normalize: (x - 0.5) / 0.5 = (img/255 - 0.5)/0.5
    train_norm = (img_np / 255.0 - 0.5) / 0.5  # = (img - 127.5) / 127.5

    diff = np.abs(mtcnn_norm - train_norm)
    print(f"  MTCNN post_process: (pixel - 127.5) / 128.0")
    print(f"  Training Normalize: (pixel - 127.5) / 127.5")
    print(f"  Max absolute diff : {diff.max():.6f}")
    print(f"  Mean absolute diff: {diff.mean():.6f}")
    print(f"  -> {'NEGLIGIBLE' if diff.max() < 0.01 else 'SIGNIFICANT'} difference")


def check_mtcnn_crops():
    """Save MTCNN crops to visually inspect."""
    print("\n" + "=" * 70)
    print("  TEST 2: MTCNN Face Crop Quality")
    print("=" * 70)

    mtcnn = MTCNN(
        image_size=160, margin=20, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        post_process=False,  # get raw [0,255] crops for saving
        device=DEVICE,
    )

    crop_dir = PROJECT_ROOT / "custom_test" / "debug_crops"
    crop_dir.mkdir(exist_ok=True)

    for img_path in IMAGES:
        img = Image.open(img_path).convert("RGB")
        face = mtcnn(img)
        if face is not None:
            # face is a tensor [3, 160, 160] in [0, 255] (post_process=False)
            face_pil = Image.fromarray(face.permute(1, 2, 0).byte().numpy())
            save_path = crop_dir / f"crop_{img_path.name}"
            face_pil.save(save_path)
            print(f"  {img_path.name}: face detected, crop saved to {save_path.name}")

            # Also get bounding box and confidence
            boxes, probs = mtcnn.detect(img)
            if boxes is not None:
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    print(f"    Face {i}: box={box.tolist()}, confidence={prob:.4f}")
        else:
            print(f"  {img_path.name}: NO FACE DETECTED")


def test_raw_pretrained():
    """Test raw VGGFace2 model (no fine-tuning) to see baseline behavior."""
    print("\n" + "=" * 70)
    print("  TEST 3: Raw Pretrained VGGFace2 (NO fine-tuning)")
    print("=" * 70)

    raw_model = load_raw_pretrained_model()
    mtcnn = MTCNN(
        image_size=160, margin=20, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        post_process=True, device=DEVICE,
    )

    embeddings = []
    labels = []
    for img_path in IMAGES:
        emb = embed_with_mtcnn(raw_model, img_path, mtcnn)
        if emb is not None:
            embeddings.append(emb)
            labels.append(img_path.stem)

    print("\n  Pairwise similarity (raw pretrained, MTCNN crop):")
    similarity_matrix(embeddings, labels)


def test_finetuned_mtcnn():
    """Test fine-tuned model with MTCNN crops."""
    print("\n" + "=" * 70)
    print("  TEST 4: Fine-tuned Model + MTCNN Crop")
    print("=" * 70)

    model = load_finetuned_model()
    mtcnn = MTCNN(
        image_size=160, margin=20, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        post_process=True, device=DEVICE,
    )

    embeddings = []
    labels = []
    for img_path in IMAGES:
        emb = embed_with_mtcnn(model, img_path, mtcnn)
        if emb is not None:
            embeddings.append(emb)
            labels.append(img_path.stem)

    print("\n  Pairwise similarity (fine-tuned, MTCNN crop):")
    similarity_matrix(embeddings, labels)


def test_finetuned_resize():
    """Test fine-tuned model with just resize (what training used)."""
    print("\n" + "=" * 70)
    print("  TEST 5: Fine-tuned Model + Resize Only (training-style)")
    print("=" * 70)

    model = load_finetuned_model()
    transform = get_default_transforms(train=False)

    embeddings = []
    labels = []
    for img_path in IMAGES:
        emb = embed_with_resize(model, img_path, transform)
        embeddings.append(emb)
        labels.append(img_path.stem)

    print("\n  Pairwise similarity (fine-tuned, resize only):")
    similarity_matrix(embeddings, labels)


def test_embedding_stats():
    """Analyze embedding distributions to check for collapse."""
    print("\n" + "=" * 70)
    print("  TEST 6: Embedding Statistics (collapse detection)")
    print("=" * 70)

    model = load_finetuned_model()
    mtcnn = MTCNN(
        image_size=160, margin=20, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        post_process=True, device=DEVICE,
    )

    embeddings = []
    for img_path in IMAGES:
        emb = embed_with_mtcnn(model, img_path, mtcnn)
        if emb is not None:
            embeddings.append(emb)

    stacked = torch.stack(embeddings)  # [N, 512]
    mean_emb = stacked.mean(dim=0)
    std_emb = stacked.std(dim=0)

    print(f"  Embedding dim   : {stacked.shape[1]}")
    print(f"  Num images      : {stacked.shape[0]}")
    print(f"  Mean L2 norm    : {torch.norm(stacked, dim=1).mean():.6f}")
    print(f"  Std of means    : {std_emb.mean():.6f}")

    # Check how many dimensions have near-zero variance
    low_var_dims = (std_emb < 0.01).sum().item()
    print(f"  Low-var dims    : {low_var_dims}/{stacked.shape[1]} (std < 0.01)")

    # Check effective dimensionality
    centered = stacked - mean_emb
    cov = centered.T @ centered / (stacked.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues.clamp(min=0)  # numerical stability
    total_var = eigenvalues.sum()
    cumsum = eigenvalues.flip(0).cumsum(0) / total_var
    effective_rank = (cumsum < 0.95).sum().item() + 1
    print(f"  Effective rank  : {effective_rank}/512 (dims for 95% variance)")

    # Per-dimension analysis: are most values the same across images?
    top_dims = std_emb.topk(5, largest=True)
    print(f"\n  Top 5 most-variable dimensions:")
    for idx, std_val in zip(top_dims.indices, top_dims.values):
        vals = stacked[:, idx].tolist()
        print(f"    dim {idx:>3}: std={std_val:.4f}, values={[f'{v:.3f}' for v in vals]}")

    bottom_dims = std_emb.topk(5, largest=False)
    print(f"\n  Top 5 least-variable dimensions:")
    for idx, std_val in zip(bottom_dims.indices, bottom_dims.values):
        vals = stacked[:, idx].tolist()
        print(f"    dim {idx:>3}: std={std_val:.4f}, values={[f'{v:.3f}' for v in vals]}")


def check_training_data():
    """Check what the training data looks like."""
    print("\n" + "=" * 70)
    print("  TEST 7: Training Data Characteristics")
    print("=" * 70)

    celeb_dir = PROJECT_ROOT / "celebs" / "Celebrity Faces Dataset"
    if not celeb_dir.exists():
        print(f"  Training data dir not found: {celeb_dir}")
        return

    person_dirs = sorted(d for d in celeb_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    print(f"  Number of classes: {len(person_dirs)}")

    for pdir in person_dirs[:3]:  # Show first 3
        images = [f for f in pdir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if images:
            img = Image.open(images[0])
            print(f"  {pdir.name}: {len(images)} images, sample size={img.size}")


def main():
    print("DEEP DIAGNOSTIC: Face Verification Similarity Investigation")
    print("Images: 1=person_A, 2=person_B, 3=person_C, 4=person_C")
    print(f"Device: {DEVICE}")

    check_preprocessing_mismatch()
    check_mtcnn_crops()
    test_raw_pretrained()
    test_finetuned_mtcnn()
    test_finetuned_resize()
    test_embedding_stats()
    check_training_data()

    print("\n" + "=" * 70)
    print("  DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
