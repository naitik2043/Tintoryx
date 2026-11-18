import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import matplotlib.pyplot as plt
from config import CFG
from src.data import make_loaders
from src.models import UNetGenerator

def load_model(path):
    model = UNetGenerator().to(CFG.device)
    ckpt = torch.load(path, map_location=CFG.device, weights_only=False)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model

@torch.no_grad()
def main():
    _, loader = make_loaders(CFG)
    gray, rgb = next(iter(loader))
    gray, rgb = gray.to(CFG.device), rgb.to(CFG.device)
    base = load_model(os.path.join(CFG.checkpoint_dir, "baseline_unet.pt"))
    gan = load_model(os.path.join(CFG.checkpoint_dir, "tintoryx_cgan_generator.pt"))
    base_out, gan_out = base(gray), gan(gray)
    n = min(8, gray.size(0))
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.5*n))
    if n == 1: axes = axes[None, :]
    titles = ["Grayscale Input", "L1-only Output", "CGAN Colorized", "Ground Truth RGB"]
    for j, title in enumerate(titles): axes[0, j].set_title(title)
    for i in range(n):
        imgs = [gray[i].repeat(3,1,1), base_out[i], gan_out[i], rgb[i]]
        for j, image in enumerate(imgs):
            axes[i,j].imshow(image.detach().cpu().permute(1,2,0).clamp(0,1))
            axes[i,j].axis("off")
    plt.tight_layout()
    os.makedirs(CFG.output_dir, exist_ok=True)
    out = os.path.join(CFG.output_dir, "tintoryx_restoration_strip.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved comparison strip: {out}")
    plt.show()
if __name__ == "__main__": main()
