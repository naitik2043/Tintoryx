"""Command-line colorization with pretrained DDColor or the academic cGAN."""

import argparse
import os
from pathlib import Path

from PIL import Image, ImageOps

from config import CFG
from src.ddcolor import DDColorColorizer


def colorize_with_cgan(image: Image.Image, checkpoint_path: str):
    """Run the original CIFAR-10 U-Net cGAN checkpoint (32x32 experiment)."""
    import numpy as np
    import torch
    from src.models import UNetGenerator

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. "
            "Train the cGAN with: python -m src.train --model cgan"
        )

    device = CFG.device
    model = UNetGenerator().to(device)
    state = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state["model"])
    model.eval()

    grayscale = ImageOps.exif_transpose(image).convert("L")
    small = grayscale.resize(
        (CFG.image_size, CFG.image_size),
        Image.Resampling.BILINEAR,
    )
    values = np.asarray(small, dtype=np.float32) / 255.0
    tensor = (
        torch.from_numpy(values)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    with torch.inference_mode():
        prediction = model(tensor).squeeze(0).clamp(0, 1)
    pixels = (
        prediction.permute(1, 2, 0).cpu().numpy() * 255.0
    ).round().astype(np.uint8)
    result = Image.fromarray(pixels, mode="RGB")
    return result.resize(grayscale.size, Image.Resampling.BICUBIC)


def main():
    parser = argparse.ArgumentParser(
        description="Colorize grayscale photographs with Tintoryx."
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument(
        "--output",
        default="outputs/colorized_image.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--model",
        choices=["ddcolor", "cgan"],
        default="ddcolor",
        help="Use pretrained DDColor (recommended) or the 32x32 academic cGAN.",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("TINTORYX_DDCOLOR_MODEL"),
        help="Optional local DDColor ONNX path.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(
            CFG.checkpoint_dir,
            "tintoryx_cgan_generator.pt",
        ),
        help="Checkpoint path when --model cgan is selected.",
    )
    args = parser.parse_args()

    with Image.open(args.input) as source:
        source.load()
        image = ImageOps.exif_transpose(source).convert("RGB")

    if args.model == "ddcolor":
        engine = DDColorColorizer(model_path=args.model_path)
        result = engine.colorize(image)
    else:
        result = colorize_with_cgan(image, args.checkpoint)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Colorized image saved to: {output_path.resolve()}")
    print(
        "Engine: "
        f"{'DDColor-tiny (512x512 chroma)' if args.model == 'ddcolor' else 'CIFAR-10 cGAN (32x32)'}"
    )


if __name__ == "__main__":
    main()
