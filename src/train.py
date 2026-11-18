"""Train the L1 U-Net baseline and/or the conditional GAN."""

import argparse
import os
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import CFG
from src.data import make_loaders
from src.models import PatchDiscriminator, UNetGenerator
from src.training_artifacts import save_training_artifacts


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and CUDA for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(path, model, epoch, optimizer=None):
    """Save model weights and optional optimizer state."""
    payload = {"epoch": epoch, "model": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def train_baseline(loader, cfg):
    generator = UNetGenerator().to(cfg.device)
    optimizer = Adam(
        generator.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, 0.999),
    )
    l1 = nn.L1Loss()
    history = []

    for epoch in range(cfg.epochs):
        generator.train()
        total_loss = 0.0
        bar = tqdm(loader, desc=f"U-Net baseline {epoch + 1}/{cfg.epochs}")

        for grayscale, real_rgb in bar:
            grayscale = grayscale.to(cfg.device, non_blocking=True)
            real_rgb = real_rgb.to(cfg.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            prediction = generator(grayscale)
            loss = l1(prediction, real_rgb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            bar.set_postfix(l1=f"{loss.item():.4f}")

        mean_loss = total_loss / max(len(loader), 1)
        history.append({"epoch": epoch + 1, "l1_loss": mean_loss})
        print(f"Baseline epoch {epoch + 1}: mean L1={mean_loss:.4f}")

    save_checkpoint(
        os.path.join(cfg.checkpoint_dir, "baseline_unet.pt"),
        generator,
        cfg.epochs,
        optimizer,
    )
    return generator, history


def train_cgan(loader, cfg):
    generator = UNetGenerator().to(cfg.device)
    discriminator = PatchDiscriminator().to(cfg.device)

    optimizer_g = Adam(
        generator.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, 0.999),
    )
    optimizer_d = Adam(
        discriminator.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, 0.999),
    )

    adversarial = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    history = []

    for epoch in range(cfg.epochs):
        generator.train()
        discriminator.train()
        totals = {
            "generator_total_loss": 0.0,
            "generator_adversarial_loss": 0.0,
            "generator_l1_loss": 0.0,
            "discriminator_loss": 0.0,
        }
        bar = tqdm(loader, desc=f"Conditional GAN {epoch + 1}/{cfg.epochs}")

        for grayscale, real_rgb in bar:
            grayscale = grayscale.to(cfg.device, non_blocking=True)
            real_rgb = real_rgb.to(cfg.device, non_blocking=True)

            # Update the discriminator using real and detached generated pairs.
            with torch.no_grad():
                fake_rgb = generator(grayscale)

            optimizer_d.zero_grad(set_to_none=True)
            real_logits = discriminator(grayscale, real_rgb)
            fake_logits = discriminator(grayscale, fake_rgb)
            discriminator_loss = 0.5 * (
                adversarial(real_logits, torch.ones_like(real_logits))
                + adversarial(fake_logits, torch.zeros_like(fake_logits))
            )
            discriminator_loss.backward()
            optimizer_d.step()

            # Update the generator while keeping discriminator weights fixed.
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)

            optimizer_g.zero_grad(set_to_none=True)
            generated_rgb = generator(grayscale)
            generated_logits = discriminator(grayscale, generated_rgb)
            generator_adversarial_loss = adversarial(
                generated_logits, torch.ones_like(generated_logits)
            )
            generator_l1_loss = l1(generated_rgb, real_rgb)
            generator_total_loss = (
                generator_adversarial_loss + cfg.lambda_l1 * generator_l1_loss
            )
            generator_total_loss.backward()
            optimizer_g.step()

            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)

            totals["generator_total_loss"] += generator_total_loss.item()
            totals["generator_adversarial_loss"] += (
                generator_adversarial_loss.item()
            )
            totals["generator_l1_loss"] += generator_l1_loss.item()
            totals["discriminator_loss"] += discriminator_loss.item()

            bar.set_postfix(
                D=f"{discriminator_loss.item():.3f}",
                G=f"{generator_total_loss.item():.3f}",
                L1=f"{generator_l1_loss.item():.3f}",
            )

        divisor = max(len(loader), 1)
        epoch_metrics = {
            key: value / divisor for key, value in totals.items()
        }
        epoch_metrics["epoch"] = epoch + 1
        history.append(epoch_metrics)
        print(
            f"CGAN epoch {epoch + 1}: "
            f"G={epoch_metrics['generator_total_loss']:.4f}, "
            f"D={epoch_metrics['discriminator_loss']:.4f}"
        )

    save_checkpoint(
        os.path.join(cfg.checkpoint_dir, "tintoryx_cgan_generator.pt"),
        generator,
        cfg.epochs,
        optimizer_g,
    )
    save_checkpoint(
        os.path.join(cfg.checkpoint_dir, "tintoryx_cgan_discriminator.pt"),
        discriminator,
        cfg.epochs,
        optimizer_d,
    )
    return generator, history


def main():
    parser = argparse.ArgumentParser(
        description="Train Tintoryx image-colorization models."
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "cgan", "both"],
        default="both",
        help="Which model to train (default: both).",
    )
    args = parser.parse_args()

    seed_everything(CFG.seed)
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)
    train_loader, _ = make_loaders(CFG)
    history = {}

    if args.model in ("baseline", "both"):
        _, history["baseline"] = train_baseline(train_loader, CFG)

    if args.model in ("cgan", "both"):
        _, history["cgan"] = train_cgan(train_loader, CFG)

    history_path, plot_path = save_training_artifacts(history, CFG)
    print(f"Saved training history: {history_path}")
    print(f"Saved training plot: {plot_path}")


if __name__ == "__main__":
    main()
