"""Minimal Streamlit interface for the Tintoryx grayscale colorizer."""

from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

from config import CFG
from src.models import UNetGenerator


CHECKPOINT_PATH = Path(CFG.checkpoint_dir) / "tintoryx_cgan_generator.pt"


@st.cache_resource(show_spinner="Loading the Tintoryx generator...")
def load_generator(checkpoint_path: str, device_name: str):
    device = torch.device(device_name)
    model = UNetGenerator().to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def predict_colorization(image: Image.Image, model, device: torch.device):
    grayscale = ImageOps.exif_transpose(image).convert("L")
    model_input = grayscale.resize(
        (CFG.image_size, CFG.image_size), Image.Resampling.BILINEAR
    )
    image_array = np.asarray(model_input, dtype=np.float32) / 255.0
    tensor = (
        torch.from_numpy(image_array)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():
        prediction = model(tensor).squeeze(0).clamp(0, 1)

    rgb_array = (
        prediction.permute(1, 2, 0).cpu().numpy() * 255.0
    ).round().astype(np.uint8)
    result = Image.fromarray(rgb_array, mode="RGB")
    return result.resize(grayscale.size, Image.Resampling.BICUBIC), grayscale


def main():
    st.set_page_config(
        page_title="Tintoryx | Image Colorization",
        page_icon="🎨",
        layout="wide",
    )
    st.title("Tintoryx")
    st.caption("Deep-learning grayscale-to-RGB colorization")
    st.write(
        "Upload a grayscale image to generate a colorized preview using "
        "the trained Conditional GAN."
    )

    st.info(
        "Research prototype: the current model was trained on 32 × 32 "
        "CIFAR-10 images. Uploaded images are reduced to 32 × 32 for "
        "prediction and the result is resized to the original dimensions. "
        "This is not high-resolution historical-photo restoration."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )
    if uploaded_file is None:
        st.stop()

    if not CHECKPOINT_PATH.is_file():
        st.error(
            "Trained CGAN checkpoint not found. Run "
            "python -m src.train --model cgan before starting the app."
        )
        st.stop()

    uploaded_image = Image.open(uploaded_file)
    device = CFG.device
    model = load_generator(str(CHECKPOINT_PATH), str(device))

    with st.spinner("Colorizing image..."):
        colorized, grayscale = predict_colorization(
            uploaded_image, model, device
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Grayscale input")
        st.image(grayscale, use_container_width=True)
    with right:
        st.subheader("Tintoryx output")
        st.image(colorized, use_container_width=True)

    buffer = BytesIO()
    colorized.save(buffer, format="PNG")
    st.download_button(
        "Download colorized image",
        data=buffer.getvalue(),
        file_name="tintoryx_colorized.png",
        mime="image/png",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
