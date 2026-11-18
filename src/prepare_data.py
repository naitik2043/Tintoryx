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
    fig, axes = plt.subplots(n, 2, figsize=(7, 2*n))

    if n == 1:
        axes = axes[None, :]

    axes[0, 0].set_title("Grayscale Input")
    axes[0, 1].set_title("Original RGB")

    for i in range(n):
        axes[i, 0].imshow(gray[i, 0], cmap="gray")
        axes[i, 1].imshow(rgb[i].permute(1, 2, 0))
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")

    os.makedirs(CFG.output_dir, exist_ok=True)

    path = os.path.join(
        CFG.output_dir,
        "dataset_pair_preview.png"
    )

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.show()

    print("Saved:", path)

if __name__ == "__main__":
    main()
