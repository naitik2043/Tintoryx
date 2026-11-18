import os

import matplotlib.pyplot as plt

from config import CFG
from src.data import make_loaders


def main():
    print("Preparing CIFAR-10 dataset...")
    train_loader, test_loader = make_loaders(CFG)
    gray, rgb = next(iter(train_loader))

    print("Training samples:", len(train_loader.dataset))
    print("Testing samples:", len(test_loader.dataset))
    print("Grayscale shape:", gray.shape)
    print("RGB shape:", rgb.shape)

    n = min(8, len(gray))
    fig, axes = plt.subplots(n, 2, figsize=(7, 2 * n))
    if n == 1:
        axes = axes[None, :]

    axes[0, 0].set_title("Grayscale Input")
    axes[0, 1].set_title("Original RGB")

    for index in range(n):
        axes[index, 0].imshow(gray[index, 0], cmap="gray")
        axes[index, 1].imshow(rgb[index].permute(1, 2, 0))
        axes[index, 0].axis("off")
        axes[index, 1].axis("off")

    os.makedirs(CFG.output_dir, exist_ok=True)
    output_path = os.path.join(CFG.output_dir, "dataset_pair_preview.png")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
