"""Run a trained Tintoryx generator on an arbitrary grayscale image."""

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageOps

from config import CFG
from src.models import UNetGenerator


def load_generator(checkpoint_path: str, device: torch.device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train the CGAN first with: python -m src.train --model cgan"
        )

    model = UNetGenerator().to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.inference_mode()
def colorize_image(
    input_path: str,
    output_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> str:
    """Colorize an image through the 32x32 model and resize the result back.

    The resize step preserves the uploaded image dimensions, but it does not
    add high-resolution restoration detail.
    """
    model = load_generator(checkpoint_path, device)

    with Image.open(input_path) as source:
        grayscale = ImageOps.exif_transpose(source).convert("L")

    original_size = grayscale.size
    model_input = grayscale.resize(
        (CFG.image_size, CFG.image_size), Image.Resampling.BILINEAR
    )
    image_array = np.asarray(model_input, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).to(device)

    prediction = model(tensor).squeeze(0).clamp(0, 1)
    rgb_array = (
        prediction.permute(1, 2, 0).cpu().numpy() * 255.0
    ).round().astype(np.uint8)
    colorized = Image.fromarray(rgb_array, mode="RGB")
    colorized = colorized.resize(original_size, Image.Resampling.BICUBIC)

    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    colorized.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Colorize a grayscale image with the Tintoryx CGAN."
    )
    parser.add_argument(
        "--input", required=True, help="Path to a grayscale image."
    )
    parser.add_argument(
        "--output",
        default="outputs/colorized_image.png",
        help="Where to save the colorized image.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(
            CFG.checkpoint_dir, "tintoryx_cgan_generator.pt"
        ),
        help="Path to a trained CGAN generator checkpoint.",
    )
    args = parser.parse_args()

    saved_path = colorize_image(
        args.input, args.output, args.checkpoint, CFG.device
    )
    print(f"Colorized image saved to: {saved_path}")
    print(
        "Note: the model operates at 32x32 resolution; the saved image is "
        "resized to the input dimensions for convenience."
    )


if __name__ == "__main__":
    main()
