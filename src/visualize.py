import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import matplotlib.pyplot as plt

from config import CFG
from src.data import make_loaders
from src.models import UNetGenerator


def load_model(path):
    model = UNetGenerator().to(CFG.device)
    checkpoint = torch.load(path, map_location=CFG.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def main():
    _, loader = make_loaders(CFG)
    gray, rgb = next(iter(loader))
    gray = gray.to(CFG.device)
    rgb = rgb.to(CFG.device)

    baseline = load_model(os.path.join(CFG.checkpoint_dir, "baseline_unet.pt"))
    cgan = load_model(
        os.path.join(CFG.checkpoint_dir, "tintoryx_cgan_generator.pt")
    )

    baseline_output = baseline(gray)
    cgan_output = cgan(gray)

    n = min(8, gray.size(0))
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.5 * n))
    if n == 1:
        axes = axes[None, :]

    titles = [
        "Grayscale Input",
        "L1-only Output",
        "CGAN Colorized",
        "Ground Truth RGB",
    ]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)

    for row in range(n):
        images = [
            gray[row].repeat(3, 1, 1),
            baseline_output[row],
            cgan_output[row],
            rgb[row],
        ]
        for column, image in enumerate(images):
            axes[row, column].imshow(
                image.detach().cpu().permute(1, 2, 0).clamp(0, 1)
            )
            axes[row, column].axis("off")

    plt.tight_layout()
    os.makedirs(CFG.output_dir, exist_ok=True)
    output_path = os.path.join(
        CFG.output_dir,
        "tintoryx_restoration_strip.png",
    )
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison strip: {output_path}")


if __name__ == "__main__":
    main()
