"""Deterministic paired CIFAR-10 data loading for image colorization."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import rgb_to_grayscale


class PairedCIFAR10(Dataset):
    """Return aligned grayscale [1,32,32] and RGB [3,32,32] tensors."""

    def __init__(self, base_dataset, indices=None):
        self.base = base_dataset
        self.indices = indices if indices is not None else range(len(base_dataset))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _ = self.base[self.indices[index]]
        grayscale = rgb_to_grayscale(image, num_output_channels=1)
        return grayscale, image


def _seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside each DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loaders(cfg):
    """Build deterministic train/test loaders from the configured subsets."""
    train_base = CIFAR10(
        cfg.data_dir, train=True, download=True, transform=ToTensor()
    )
    test_base = CIFAR10(
        cfg.data_dir, train=False, download=True, transform=ToTensor()
    )

    rng = random.Random(cfg.seed)
    train_ids = list(range(len(train_base)))
    test_ids = list(range(len(test_base)))
    rng.shuffle(train_ids)
    rng.shuffle(test_ids)

    train_ds = PairedCIFAR10(train_base, train_ids[: cfg.train_subset])
    test_ds = PairedCIFAR10(test_base, test_ids[: cfg.test_subset])

    train_generator = torch.Generator()
    train_generator.manual_seed(cfg.seed)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
    )
    return train_loader, test_loader
