"""Tintoryx web app: pretrained DDColor photo inference + academic baseline."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from config import CFG
from src.ddcolor import DDColorColorizer

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
UNET_CHECKPOINT = Path(CFG.checkpoint_dir) / "tintoryx_cgan_generator.pt"


@st.cache_resource(show_spinner=False)
def load_ddcolor(model_path: str | None):
    return DDColorColorizer(
        model_path=model_path,
        max_image_pixels=MAX_IMAGE_PIXELS,
    )


@st.cache_resource(show_spinner=False)
def load_experimental_unet(checkpoint_path: str, device_name: str):
    import torch
    from src.models import UNetGenerator

    device = torch.device(device_name)
    model = UNetGenerator().to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def run_experimental_unet(image: Image.Image, model):
    """Run the course U-Net experiment (intentionally limited to 32x32)."""
    import numpy as np
    import torch

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
        .to(CFG.device)
    )
    with torch.inference_mode():
        prediction = model(tensor).squeeze(0).clamp(0, 1)
    pixels = (
        prediction.permute(1, 2, 0).cpu().numpy() * 255.0
    ).round().astype(np.uint8)
    result = Image.fromarray(pixels, mode="RGB")
    return result.resize(grayscale.size, Image.Resampling.BICUBIC), grayscale


def main():
    st.set_page_config(
        page_title="Tintoryx | AI Image Colorization",
        page_icon="🎨",
        layout="wide",
    )
    st.title("Tintoryx")
    st.caption("AI-powered grayscale photograph colorization")
    st.markdown(
        "Create a colorized preview from a black-and-white photograph using "
        "a pretrained image-colorization model."
    )

    engine = st.selectbox(
        "Colorization engine",
        options=[
            "DDColor — pretrained photo colorizer (recommended)",
            "Tintoryx U-Net — CIFAR-10 academic experiment",
        ],
        index=0,
    )
    use_ddcolor = engine.startswith("DDColor")

    if use_ddcolor:
        st.info(
            "DDColor-tiny predicts color at 512 × 512 while retaining the "
            "original photograph's full-resolution luminance. Predicted "
            "chroma is resized to the original image dimensions. It is much "
            "better suited to photographs than the 32 × 32 course model, "
            "but it is not a dedicated 4K restoration or scratch-repair tool. "
            "The quantized ONNX model (~108 MiB) downloads once on first use."
        )
    else:
        st.warning(
            "Academic experiment only: this U-Net was trained on 32 × 32 "
            "CIFAR-10 images. It will look blurry on real photographs and "
            "is retained for coursework comparison."
        )

    uploaded_file = st.file_uploader(
        "Upload a grayscale or black-and-white photograph",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        help="For best results, use a clear photograph with good contrast.",
    )
    if uploaded_file is None:
        st.stop()

    if uploaded_file.size > MAX_UPLOAD_BYTES:
        st.error("The upload is larger than 20 MB. Please use a smaller file.")
        st.stop()

    try:
        with Image.open(uploaded_file) as source:
            source.load()
            original = ImageOps.exif_transpose(source).convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ) as exc:
        st.error(f"Could not read this image: {exc}")
        st.stop()

    if original.width * original.height > MAX_IMAGE_PIXELS:
        st.error(
            "This image exceeds the 16-megapixel processing limit. "
            "Resize it before uploading."
        )
        st.stop()

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Original image")
        st.image(original, use_container_width=True)

    try:
        if use_ddcolor:
            with st.spinner(
                "Loading DDColor model (first run downloads ~108 MiB)..."
            ):
                colorizer = load_ddcolor(None)
            with st.spinner("Colorizing photograph..."):
                colorized = colorizer.colorize(original)
        else:
            if not UNET_CHECKPOINT.is_file():
                st.error(
                    "U-Net checkpoint not found. Train the academic model "
                    "with: python -m src.train --model cgan"
                )
                st.stop()
            import torch

            with st.spinner("Loading academic U-Net..."):
                model = load_experimental_unet(
                    str(UNET_CHECKPOINT),
                    str(CFG.device),
                )
            with st.spinner("Colorizing with the academic model..."):
                colorized, _ = run_experimental_unet(original, model)

    except Exception as exc:
        st.error(f"Colorization could not be completed: {exc}")
        st.stop()

    with right:
        st.subheader("Tintoryx colorized output")
        st.image(colorized, use_container_width=True)

    st.success(
        f"Completed at {original.width:,} × {original.height:,} pixels "
        f"using {'DDColor-tiny' if use_ddcolor else 'the CIFAR-10 U-Net'}."
    )
    output = BytesIO()
    colorized.save(output, format="PNG")
    st.download_button(
        "Download colorized image",
        data=output.getvalue(),
        file_name="tintoryx_colorized.png",
        mime="image/png",
        use_container_width=True,
    )
    st.caption(
        "AI colorization is a prediction. The original photograph's true "
        "colors cannot be guaranteed."
    )
    st.markdown(
        "Colorization model: [DDColor (ICCV 2023)]"
        "(https://github.com/piddnad/DDColor) · "
        "[Quantized ONNX model card]"
        "(https://huggingface.co/toolsmonk/ddcolor-tiny)"
    )


if __name__ == "__main__":
    main()
