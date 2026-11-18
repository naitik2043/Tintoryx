from dataclasses import dataclass
import torch

@dataclass
class Config:
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    train_subset: int = 10_000
    test_subset: int = 1_000
    batch_size: int = 16
    epochs: int = 12
    learning_rate: float = 2e-4
    beta1: float = 0.5
    lambda_l1: float = 10.0
    num_workers: int = 0
    seed: int = 42
    image_size: int = 32

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = Config()

