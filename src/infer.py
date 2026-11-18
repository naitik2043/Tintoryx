import os
import torch
import matplotlib.pyplot as plt

from config import CFG
from src.data import make_loaders
from src.models import UNetGenerator

@torch.no_grad()
def main():
    device = CFG.device

    path = os.path.join(
        CFG.checkpoint_dir,
        "tintoryx_cgan_generator.pt"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Train the original CGAN first using: "
            "python -m src.train --model cgan"
        )

    model = UNetGenerator().to(device)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    _, loader = make_loaders(CFG)
    gray, rgb = next(iter(loader))

    gray = gray.to(device)
    rgb = rgb.to(device)

    prediction = model(gray)

    n = min(8, len(gray))
    fig, axes = plt.subplots(n, 3, figsize=(10, 3*n))

    if n == 1:
        axes = axes[None, :]

    titles = ["Grayscale", "Tintoryx Prediction", "Original RGB"]

    for j, title in enumerate(titles):
        axes[0, j].set_title(title)

    for i in range(n):
        axes[i, 0].imshow(gray[i, 0].cpu(), cmap="gray")
        axes[i, 1].imshow(
            prediction[i].cpu().permute(1, 2, 0).clamp(0, 1)
        )
        axes[i, 2].imshow(rgb[i].cpu().permute(1, 2, 0))

        for j in range(3):
            axes[i, j].axis("off")

    os.makedirs(CFG.output_dir, exist_ok=True)

    output = os.path.join(
        CFG.output_dir,
        "tintoryx_trained_predictions.png"
    )

    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()

    mse = ((prediction-rgb)**2).mean(dim=(1,2,3))
    psnr = (10*torch.log10(1/mse.clamp_min(1e-10))).mean()

    print("Mean PSNR:", round(psnr.item(), 3), "dB")
    print("Saved:", output)

if __name__ == "__main__":
    main()

