import os
import sys
import argparse
import random

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from config import CFG
from src.data import make_loaders
from src.models import UNetGenerator, PatchDiscriminator


def seed_everything(seed):
    """Seed Python and PyTorch RNGs for repeatable initialization and shuffling."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, epoch, optimizer=None):
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

    for epoch in range(cfg.epochs):
        generator.train()
        total = 0.0
        bar = tqdm(loader, desc=f"L1 baseline {epoch + 1}/{cfg.epochs}")

        for gray, rgb in bar:
            gray = gray.to(cfg.device)
            rgb = rgb.to(cfg.device)

            optimizer.zero_grad(set_to_none=True)
            loss = l1(generator(gray), rgb)
            loss.backward()
            optimizer.step()

            total += loss.item()
            bar.set_postfix(l1=f"{loss.item():.4f}")

        print(f"Baseline epoch {epoch + 1}: mean L1={total / len(loader):.4f}")

    save_checkpoint(
        os.path.join(cfg.checkpoint_dir, "baseline_unet.pt"),
        generator,
        cfg.epochs,
        optimizer,
    )
    return generator


def train_cgan(loader, cfg):
    generator = UNetGenerator().to(cfg.device)
    discriminator = PatchDiscriminator().to(cfg.device)

    opt_g = Adam(
        generator.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, 0.999),
    )
    opt_d = Adam(
        discriminator.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, 0.999),
    )

    adversarial = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    for epoch in range(cfg.epochs):
        generator.train()
        discriminator.train()
        bar = tqdm(loader, desc=f"Conditional GAN {epoch + 1}/{cfg.epochs}")

        for gray, real in bar:
            gray = gray.to(cfg.device)
            real = real.to(cfg.device)

            # Train discriminator: real pairs -> 1, generated pairs -> 0.
            with torch.no_grad():
                fake = generator(gray)

            opt_d.zero_grad(set_to_none=True)
            real_logits = discriminator(gray, real)
            fake_logits = discriminator(gray, fake.detach())

            d_loss = 0.5 * (
                adversarial(real_logits, torch.ones_like(real_logits))
                + adversarial(fake_logits, torch.zeros_like(fake_logits))
            )
            d_loss.backward()
            opt_d.step()

            # Train generator to fool D and stay close to paired RGB target.
            opt_g.zero_grad(set_to_none=True)
            generated = generator(gray)
            fake_logits = discriminator(gray, generated)

            g_adv = adversarial(fake_logits, torch.ones_like(fake_logits))
            g_l1 = l1(generated, real)
            g_loss = g_adv + cfg.lambda_l1 * g_l1

            g_loss.backward()
            opt_g.step()

            bar.set_postfix(
                D=f"{d_loss.item():.3f}",
                G=f"{g_loss.item():.3f}",
                L1=f"{g_l1.item():.3f}",
            )

        print(f"CGAN epoch {epoch + 1} complete")

    save_checkpoint(
        os.path.join(cfg.checkpoint_dir, "tintoryx_cgan_generator.pt"),
        generator,
        cfg.epochs,
        opt_g,
    )
    save_checkpoint(
        os.path.join(cfg.checkpoint_dir, "tintoryx_cgan_discriminator.pt"),
        discriminator,
        cfg.epochs,
        opt_d,
    )
    return generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["baseline", "cgan", "both"],
        default="both",
    )
    args = parser.parse_args()

    seed_everything(CFG.seed)
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)
    train_loader, _ = make_loaders(CFG)

    if args.model in ("baseline", "both"):
        train_baseline(train_loader, CFG)

    if args.model in ("cgan", "both"):
        train_cgan(train_loader, CFG)


if __name__ == "__main__":
    main()
