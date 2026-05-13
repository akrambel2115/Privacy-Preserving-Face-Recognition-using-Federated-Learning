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


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class FaceDataset(Dataset):
    """A PyTorch Dataset that loads face images from a directory.

    See module docstring for the supported directory layouts.
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

        self.samples: list[tuple[Path, int]] = []
        self.class_to_idx: dict[str, int] = {}

        subdirs = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if subdirs:
            for idx, subdir in enumerate(subdirs):
                self.class_to_idx[subdir.name] = idx
                for img_path in sorted(subdir.iterdir()):
                    if img_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                        self.samples.append((img_path, idx))
        else:
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
# DataLoader builders
# ---------------------------------------------------------------------------
#
# Performance note: persistent_workers=True keeps the worker processes alive
# across epochs/rounds. Without this, every time a DataLoader is iterated
# from scratch, num_workers worker processes are spawned and torn down,
# which is a significant per-round cost when there are 1000 client loaders.
#
# We accept num_workers as an argument and only enable persistent_workers
# when num_workers > 0.

def _loader_kwargs(num_workers: int, train: bool) -> dict:
    kwargs: dict = {
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
        "shuffle": train,
        "drop_last": train,
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        # prefetch_factor only valid when num_workers > 0
        kwargs["prefetch_factor"] = 2
    return kwargs


def load_client_dataset(
    data_dir: str,
    client_id: int = 0,
    train: bool = True,
    batch_size: int = 16,
    num_workers: int = 0,
) -> DataLoader:
    transform = get_default_transforms(train=train)
    dataset = FaceDataset(root_dir=data_dir, client_id=client_id, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, **_loader_kwargs(num_workers, train))


def load_full_dataset(
    data_dir: str,
    train: bool = True,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    transform = get_default_transforms(train=train)
    dataset = FaceDataset(root_dir=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, **_loader_kwargs(num_workers, train))


def partition_dataset_by_client(
    data_dir: str,
    train: bool = True,
    batch_size: int = 16,
    num_workers: int = 0,
) -> dict[int, DataLoader]:
    """Partition a multi-person dataset into per-client positive-only loaders.

    Note: at large client counts (e.g. 1000), allocating num_workers worker
    processes per client is infeasible. Recommended values:
      - num_workers=0  for >200 clients (workers spawned by clients run sequentially anyway)
      - num_workers=2-4 for <100 clients
    The fused-round path uses a separate combined loader and ignores this.
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
            **_loader_kwargs(num_workers, train),
        )

    return client_loaders


def list_client_dirs(data_dir: str) -> list[Path]:
    """Return the sorted list of per-client subdirectories. Used by fused mode."""
    root = Path(data_dir)
    return sorted([
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])


def get_num_classes(data_dir: str) -> int:
    root = Path(data_dir)
    return len([
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])