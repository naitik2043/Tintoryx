"""Image-quality metrics for tensors in NCHW format with values in [0, 1]."""

import torch
import torch.nn.functional as F


def _validate_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction and target must have the same shape; got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}."
        )
    if pred.ndim != 4:
        raise ValueError(
            f"Expected NCHW image tensors, received {pred.ndim}-D tensors."
        )


@torch.no_grad()
def batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Return mean per-image PSNR in dB for images whose data range is [0, 1].

    A small MSE floor makes the score finite for identical images (100 dB).
    """
    _validate_pair(pred, target)
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3)).clamp_min(1e-10)
    return (10.0 * torch.log10(1.0 / mse)).mean().item()


def _gaussian_window(
    channels: int,
    size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.arange(size, device=device, dtype=dtype)
    coordinates = coordinates - (size - 1) / 2
    gaussian = torch.exp(-(coordinates**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    window_2d = torch.outer(gaussian, gaussian)
    return (
        window_2d.view(1, 1, size, size)
        .expand(channels, 1, size, size)
        .contiguous()
    )


@torch.no_grad()
def batch_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Return mean per-image SSIM using a Gaussian window.

    Inputs must be NCHW tensors with the same shape. The default data range is
    1 because the colorization pipeline scales image tensors to [0, 1].
    """
    _validate_pair(pred, target)
    if data_range <= 0:
        raise ValueError("data_range must be positive.")

    height, width = pred.shape[-2:]
    size = min(window_size, height, width)
    if size % 2 == 0:
        size -= 1
    if size < 1:
        raise ValueError("Images must have non-zero height and width.")

    channels = pred.shape[1]
    window = _gaussian_window(
        channels, size, sigma, pred.device, pred.dtype
    )

    # Valid convolution avoids introducing artificial zero-padding at borders.
    mu_pred = F.conv2d(pred, window, groups=channels)
    mu_target = F.conv2d(target, window, groups=channels)

    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_cross = mu_pred * mu_target

    var_pred = F.conv2d(pred.square(), window, groups=channels) - mu_pred_sq
    var_target = (
        F.conv2d(target.square(), window, groups=channels) - mu_target_sq
    )
    covariance = (
        F.conv2d(pred * target, window, groups=channels) - mu_cross
    )

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_cross + c1) * (2 * covariance + c2)
    denominator = (
        (mu_pred_sq + mu_target_sq + c1)
        * (var_pred + var_target + c2)
    ).clamp_min(torch.finfo(pred.dtype).eps)

    score_map = numerator / denominator
    per_image = score_map.mean(dim=(1, 2, 3))
    return per_image.mean().item()
