import torch

@torch.no_grad()
def batch_psnr(pred, target):
    # Inputs are in [0,1]. Compute per-image PSNR, then average.
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3)).clamp_min(1e-10)
    return (10.0 * torch.log10(1.0 / mse)).mean().item()
