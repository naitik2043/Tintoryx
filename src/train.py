import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from config import CFG
from src.data import make_loaders
from src.models import UNetGenerator, PatchDiscriminator

def save_checkpoint(path, model, epoch, optimizer=None):
    payload = {"epoch": epoch, "model": model.state_dict()}
    if optimizer is not None: payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)

def train_baseline(loader, cfg):
    G = UNetGenerator().to(cfg.device)
    opt = Adam(G.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))
    l1 = nn.L1Loss()
    for epoch in range(cfg.epochs):
        G.train(); total = 0
        bar = tqdm(loader, desc=f"L1 baseline {epoch+1}/{cfg.epochs}")
        for gray, rgb in bar:
            gray, rgb = gray.to(cfg.device), rgb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            loss = l1(G(gray), rgb)
            loss.backward(); opt.step()
            total += loss.item()
            bar.set_postfix(l1=f"{loss.item():.4f}")
        print(f"Baseline epoch {epoch+1}: mean L1={total/len(loader):.4f}")
    save_checkpoint(os.path.join(cfg.checkpoint_dir, "baseline_unet.pt"), G, cfg.epochs, opt)
    return G

def train_cgan(loader, cfg):
    G, D = UNetGenerator().to(cfg.device), PatchDiscriminator().to(cfg.device)
    opt_g = Adam(G.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))
    opt_d = Adam(D.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))
    adv = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    for epoch in range(cfg.epochs):
        G.train(); D.train()
        bar = tqdm(loader, desc=f"Conditional GAN {epoch+1}/{cfg.epochs}")
        for gray, real in bar:
            gray, real = gray.to(cfg.device), real.to(cfg.device)
            # Train discriminator: real pairs -> 1, generated pairs -> 0.
            with torch.no_grad(): fake = G(gray)
            opt_d.zero_grad(set_to_none=True)
            real_logits = D(gray, real)
            fake_logits = D(gray, fake.detach())
            d_loss = 0.5 * (adv(real_logits, torch.ones_like(real_logits)) +
                            adv(fake_logits, torch.zeros_like(fake_logits)))
            d_loss.backward(); opt_d.step()

            # Train generator to fool D and stay close to paired RGB target.
            opt_g.zero_grad(set_to_none=True)
            generated = G(gray)
            fake_logits = D(gray, generated)
            g_adv = adv(fake_logits, torch.ones_like(fake_logits))
            g_l1 = l1(generated, real)
            g_loss = g_adv + cfg.lambda_l1 * g_l1
            g_loss.backward(); opt_g.step()
            bar.set_postfix(D=f"{d_loss.item():.3f}", G=f"{g_loss.item():.3f}",
                            L1=f"{g_l1.item():.3f}")
        print(f"CGAN epoch {epoch+1} complete")
    save_checkpoint(os.path.join(cfg.checkpoint_dir, "tintoryx_cgan_generator.pt"), G, cfg.epochs, opt_g)
    save_checkpoint(os.path.join(cfg.checkpoint_dir, "tintoryx_cgan_discriminator.pt"), D, cfg.epochs, opt_d)
    return G

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "cgan", "both"], default="both")
    args = parser.parse_args()
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)
    train_loader, _ = make_loaders(CFG)
    if args.model in ("baseline", "both"): train_baseline(train_loader, CFG)
    if args.model in ("cgan", "both"): train_cgan(train_loader, CFG)

if __name__ == "__main__": main()
