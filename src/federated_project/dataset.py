# dataset loading and processing
#
# In the federated face recognition setup each client trains using ONLY
# positive samples (their own face images).  No client ever sees another
# person's images.  The dataset utilities below are designed around this
# constraint: ``load_client_dataset`` returns a loader whose every sample
# shares the same label (the client's ID), and ``partition_dataset_by_client``
# splits a multi-person directory so that each resulting loader contains
# exclusively one person's images.

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Default transforms for FaceNet (InceptionResnetV1)
# FaceNet expects images of size 160x160 with pixel values in [-1, 1].
# ---------------------------------------------------------------------------

def get_default_transforms(train: bool = True) -> transforms.Compose:
    """
    Return the default image transforms for the FaceNet pipeline.

    Training transforms include random augmentations (horizontal flip,
    slight rotation, color jitter) to improve generalization.
    Evaluation transforms only resize, crop, and normalize.

    Args:
        train: If True, include data-augmentation transforms.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((170, 170)),
            transforms.RandomCrop(160),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


# ---------------------------------------------------------------------------
# Core Dataset class
# ---------------------------------------------------------------------------

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class FaceDataset(Dataset):
    """
    A PyTorch ``Dataset`` that loads face images from a directory.

    In the federated setting each client owns **only positive samples**
    (their own face images).  The single-person layout below is the
    primary mode used during local client training — every image in the
    folder receives the same ``client_id`` label so the positive-only
    loss (squared hinge with cosine similarity) can be applied directly.

    Expected directory layout — one of two formats:

    **Single-person / positive-only (client-local) layout** ::

        root/
            img_001.jpg
            img_002.jpg
            ...

    All images are assigned ``client_id`` as their label.  This is the
    layout each Flower client uses — it contains only that person's face
    images (positive class) and nothing else.

    **Multi-person layout** (server-side / simulation only) ::

        root/
            person_a/
                img_001.jpg
                ...
            person_b/
                img_001.jpg
                ...

    Each subdirectory is treated as a separate class.  This layout is
    used for centralized evaluation or to partition data for simulation;
    individual clients never see this combined view.

    Args:
        root_dir:   Path to the root image directory.
        client_id:  Integer label to assign in single-person mode.  This
                    corresponds to the client's row in the global W matrix.
        transform:  Optional torchvision transform pipeline.  If ``None``,
                    the default evaluation transforms are used.
    """

    def __init__(
        self,
        root_dir: str,
        client_id: int = 0,
        transform: Optional[transforms.Compose] = None,
    ):
        self.root_dir = Path(root_dir)
        self.client_id = client_id
        self.transform = transform or get_default_transforms(train=False)

        # Discover images and build (path, label) pairs
        self.samples: list[tuple[Path, int]] = []
        self.class_to_idx: dict[str, int] = {}

        subdirs = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if subdirs:
            # Multi-person layout
            for idx, subdir in enumerate(subdirs):
                self.class_to_idx[subdir.name] = idx
                for img_path in sorted(subdir.iterdir()):
                    if img_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                        self.samples.append((img_path, idx))
        else:
            # Single-person (client-local) layout
            self.class_to_idx["self"] = client_id
            for img_path in sorted(self.root_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    self.samples.append((img_path, client_id))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No valid images found in '{self.root_dir}'. "
                f"Supported extensions: {VALID_IMAGE_EXTENSIONS}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# High-level loader helpers
# ---------------------------------------------------------------------------

def load_client_dataset(
    data_dir: str,
    client_id: int = 0,
    train: bool = True,
    batch_size: int = 16,
    num_workers: int = 0,
) -> DataLoader:
    """
    Load a single client's **positive-only** face dataset.

    This is the main entry-point used by the Flower client.  It expects
    ``data_dir`` to point to a directory containing **only that client's
    own face images** (flat layout, single positive class).  Every sample
    in the returned loader shares the same label (``client_id``).

    Because training uses a positive-only loss (squared hinge with cosine
    similarity against the client's embedding row ``W[client_id]``), no
    negative / other-person images are needed or expected.

    Args:
        data_dir:    Path to the client's local data folder.
        client_id:   Numeric ID of this client — corresponds to the row
                     index in the global classification matrix ``W``.
        train:       If True, apply training augmentations.
        batch_size:  Mini-batch size.
        num_workers: Number of data-loading workers.

    Returns:
        A ``torch.utils.data.DataLoader`` over the client's images.
    """
    transform = get_default_transforms(train=train)
    dataset = FaceDataset(root_dir=data_dir, client_id=client_id, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def load_full_dataset(
    data_dir: str,
    train: bool = True,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    """
    Load the full multi-person dataset and return a ``DataLoader``.

    This is useful for centralized pre-training or evaluation on the
    server side.  It expects the multi-person directory layout where each
    subdirectory is a different person.

    Args:
        data_dir:    Path to the root dataset folder (e.g. ``data/celebs/Celebrity Faces Dataset``).
        train:       If True, apply training augmentations.
        batch_size:  Mini-batch size.
        num_workers: Number of data-loading workers.

    Returns:
        A ``torch.utils.data.DataLoader`` over the full dataset.
    """
    transform = get_default_transforms(train=train)
    dataset = FaceDataset(root_dir=data_dir, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def partition_dataset_by_client(
    data_dir: str,
    train: bool = True,
    batch_size: int = 16,
    num_workers: int = 0,
) -> dict[int, DataLoader]:
    """
    Partition a multi-person dataset into per-client positive-only loaders.

    Simulates the federated setting: each subdirectory in ``data_dir``
    becomes a separate client whose ``DataLoader`` contains **only that
    person's images** (positive class).  No client loader will ever
    include images from another person.

    Args:
        data_dir:    Path to the root dataset folder (multi-person layout).
        train:       If True, apply training augmentations.
        batch_size:  Mini-batch size per client.
        num_workers: Number of data-loading workers per loader.

    Returns:
        A dict mapping ``client_id`` → positive-only ``DataLoader``.
    """
    root = Path(data_dir)
    transform = get_default_transforms(train=train)

    subdirs = sorted([
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not subdirs:
        raise FileNotFoundError(
            f"No person subdirectories found under '{data_dir}'. "
            "Expected a multi-person directory layout."
        )

    client_loaders: dict[int, DataLoader] = {}
    for client_id, person_dir in enumerate(subdirs):
        dataset = FaceDataset(
            root_dir=str(person_dir),
            client_id=client_id,
            transform=transform,
        )
        client_loaders[client_id] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=train,
        )

    return client_loaders


def get_num_classes(data_dir: str) -> int:
    """
    Count the number of person (class) subdirectories in a dataset root.

    Args:
        data_dir: Path to the root dataset folder.

    Returns:
        Number of person subdirectories found.
    """
    root = Path(data_dir)
    return len([
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])