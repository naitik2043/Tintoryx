import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from config import CFG
from src.data import make_loaders
from src.models import UNetGenerator
from src.metrics import batch_psnr

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); total_psnr = 0.0; count = 0
    for gray, rgb in loader:
        gray, rgb = gray.to(device), rgb.to(device)
        pred = model(gray)
        batch_n = gray.size(0)
        total_psnr += batch_psnr(pred, rgb) * batch_n
        count += batch_n
    return total_psnr / count

def load_generator(path):
    model = UNetGenerator().to(CFG.device)
    checkpoint = torch.load(path, map_location=CFG.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model

def main():
    _, test_loader = make_loaders(CFG)
    base_path = os.path.join(CFG.checkpoint_dir, "baseline_unet.pt")
    gan_path = os.path.join(CFG.checkpoint_dir, "tintoryx_cgan_generator.pt")
    baseline = load_generator(base_path)
    cgan = load_generator(gan_path)
    p1 = evaluate(baseline, test_loader, CFG.device)
    p2 = evaluate(cgan, test_loader, CFG.device)
    print("\n========== TEST RESULTS ==========")
    print(f"L1-only U-Net PSNR : {p1:.3f} dB")
    print(f"CGAN U-Net PSNR    : {p2:.3f} dB")
    print(f"Difference         : {p2-p1:+.3f} dB")
    print("Note: higher PSNR means lower pixel-wise reconstruction error.")

    import matplotlib.pyplot as plt
    os.makedirs(CFG.output_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    bars = plt.bar(["L1-only U-Net", "Tintoryx CGAN"], [p1, p2])
    plt.ylabel("Average Test PSNR (dB)")
    plt.title("CIFAR-10 Colorization: PSNR Comparison")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, [p1, p2]):
        plt.text(bar.get_x() + bar.get_width()/2, value, f"{value:.2f}",
                 ha="center", va="bottom")
    plt.tight_layout()
    chart_path = os.path.join(CFG.output_dir, "psnr_comparison.png")
    plt.savefig(chart_path, dpi=180, bbox_inches="tight")
    print(f"Saved PSNR chart: {chart_path}")
if __name__ == "__main__": main()
