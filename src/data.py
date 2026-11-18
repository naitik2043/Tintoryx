import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import rgb_to_grayscale
import random

class PairedCIFAR10(Dataset):
    """Returns paired tensors: grayscale [1,32,32], RGB [3,32,32], in [0,1]."""
    def __init__(self, base_dataset, indices=None):
        self.base = base_dataset
        self.indices = indices if indices is not None else range(len(base_dataset))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _ = self.base[self.indices[index]]
        gray = rgb_to_grayscale(image, num_output_channels=1)
        return gray, image

def make_loaders(cfg):
    train_base = CIFAR10(cfg.data_dir, train=True, download=True, transform=ToTensor())
    test_base = CIFAR10(cfg.data_dir, train=False, download=True, transform=ToTensor())

    rng = random.Random(cfg.seed)
    train_ids = list(range(len(train_base)))
    test_ids = list(range(len(test_base)))
    rng.shuffle(train_ids); rng.shuffle(test_ids)
    train_ids = train_ids[:cfg.train_subset]
    test_ids = test_ids[:cfg.test_subset]

    train_ds = PairedCIFAR10(train_base, train_ids)
    test_ds = PairedCIFAR10(test_base, test_ids)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, test_loader
