"""Persist and plot epoch-level training metrics."""

import json
import os
from dataclasses import asdict

import matplotlib.pyplot as plt


def save_training_artifacts(history: dict, cfg) -> tuple[str, str]:
    """Save training history as JSON and create a training-loss figure."""
    os.makedirs(cfg.output_dir, exist_ok=True)

    history_path = os.path.join(cfg.output_dir, "training_history.json")
    plot_path = os.path.join(cfg.output_dir, "training_history.png")
    payload = {
        "configuration": asdict(cfg),
        "history": history,
    }
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    has_baseline = bool(history.get("baseline"))
    has_cgan = bool(history.get("cgan"))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if has_baseline:
        epochs = [item["epoch"] for item in history["baseline"]]
        losses = [item["l1_loss"] for item in history["baseline"]]
        axes[0].plot(epochs, losses, marker="o", label="U-Net L1 loss")
        axes[0].set_title("U-Net baseline")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Mean L1 loss")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
    else:
        axes[0].text(0.5, 0.5, "Baseline not trained in this run",
                     ha="center", va="center")
        axes[0].set_axis_off()

    if has_cgan:
        epochs = [item["epoch"] for item in history["cgan"]]
        axes[1].plot(
            epochs,
            [item["generator_total_loss"] for item in history["cgan"]],
            marker="o",
            label="Generator total",
        )
        axes[1].plot(
            epochs,
            [item["discriminator_loss"] for item in history["cgan"]],
            marker="s",
            label="Discriminator",
        )
        axes[1].set_title("Conditional GAN")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Mean loss")
        axes[1].legend()
        axes[1].grid(alpha=0.25)
    else:
        axes[1].text(0.5, 0.5, "Conditional GAN not trained in this run",
                     ha="center", va="center")
        axes[1].set_axis_off()

    figure.suptitle("Tintoryx training history", fontsize=14)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return history_path, plot_path
