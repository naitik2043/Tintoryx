"""Evaluate trained generators on the held-out CIFAR-10 subset."""

import json
import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import CFG
from src.data import make_loaders
from src.metrics import batch_psnr, batch_ssim
from src.models import UNetGenerator


@torch.no_grad()
def evaluate(model, loader, device):
    """Return sample-weighted mean PSNR and SSIM for a data loader."""
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for grayscale, real_rgb in loader:
        grayscale = grayscale.to(device, non_blocking=True)
        real_rgb = real_rgb.to(device, non_blocking=True)
        prediction = model(grayscale)
        batch_size = grayscale.size(0)

        total_psnr += batch_psnr(prediction, real_rgb) * batch_size
        total_ssim += batch_ssim(prediction, real_rgb) * batch_size
        count += batch_size

    if count == 0:
        raise ValueError("The evaluation data loader is empty.")

    return {
        "psnr_db": total_psnr / count,
        "ssim": total_ssim / count,
        "samples": count,
    }


def load_generator(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. "
            "Train the models first with: python -m src.train --model both"
        )

    model = UNetGenerator().to(CFG.device)
    checkpoint = torch.load(
        path, map_location=CFG.device, weights_only=True
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def save_metrics_chart(results, output_path):
    labels = ["L1-only U-Net", "Conditional GAN"]
    psnr_values = [
        results["baseline"]["psnr_db"],
        results["cgan"]["psnr_db"],
    ]
    ssim_values = [
        results["baseline"]["ssim"],
        results["cgan"]["ssim"],
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, values, title, ylabel in (
        (axes[0], psnr_values, "PSNR comparison", "PSNR (dB)"),
        (axes[1], ssim_values, "SSIM comparison", "SSIM"),
    ):
        bars = axis.bar(labels, values)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=10)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

    figure.suptitle("Tintoryx image-quality evaluation", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    _, test_loader = make_loaders(CFG)
    baseline_path = os.path.join(
        CFG.checkpoint_dir, "baseline_unet.pt"
    )
    cgan_path = os.path.join(
        CFG.checkpoint_dir, "tintoryx_cgan_generator.pt"
    )

    baseline = load_generator(baseline_path)
    cgan = load_generator(cgan_path)

    results = {
        "dataset": "CIFAR-10",
        "split": "deterministic test subset",
        "configuration": {
            "seed": CFG.seed,
            "test_subset": CFG.test_subset,
            "image_size": CFG.image_size,
        },
        "baseline": evaluate(baseline, test_loader, CFG.device),
        "cgan": evaluate(cgan, test_loader, CFG.device),
    }
    results["cgan_minus_baseline"] = {
        "psnr_db": (
            results["cgan"]["psnr_db"] - results["baseline"]["psnr_db"]
        ),
        "ssim": results["cgan"]["ssim"] - results["baseline"]["ssim"],
    }

    os.makedirs(CFG.output_dir, exist_ok=True)
    json_path = os.path.join(CFG.output_dir, "evaluation_results.json")
    chart_path = os.path.join(CFG.output_dir, "metrics_comparison.png")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    save_metrics_chart(results, chart_path)

    print("\n========== TEST RESULTS ==========")
    for name, label in (
        ("baseline", "L1-only U-Net"),
        ("cgan", "Conditional GAN"),
    ):
        metric = results[name]
        print(
            f"{label:18s} PSNR: {metric['psnr_db']:.3f} dB | "
            f"SSIM: {metric['ssim']:.4f}"
        )
    print(
        "CGAN - baseline   "
        f"PSNR: {results['cgan_minus_baseline']['psnr_db']:+.3f} dB | "
        f"SSIM: {results['cgan_minus_baseline']['ssim']:+.4f}"
    )
    print(f"Saved metrics: {json_path}")
    print(f"Saved comparison chart: {chart_path}")


if __name__ == "__main__":
    main()
