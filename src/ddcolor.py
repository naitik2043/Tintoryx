"""High-resolution grayscale colorization with a pretrained DDColor-tiny ONNX model.

DDColor predicts CIE Lab chroma at 512x512. Tintoryx retains the source
photograph's original-resolution luminance and combines it with the predicted
chroma after resizing the chroma planes to the source dimensions.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

DDCOLOR_REPO_ID = "toolsmonk/ddcolor-tiny"
DDCOLOR_REVISION = "c7d348d"
DDCOLOR_FILENAME = "ddcolor_tiny_512_int8.onnx"
DDCOLOR_SHA256 = "e983347272b8e218edc974a0af2bd66f53b73212abb800e103c64806e4c14a14"
DDCOLOR_INPUT_SIZE = 512
DEFAULT_MAX_IMAGE_PIXELS = 16_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_path(
    model_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a local model, or download and verify the pinned public ONNX.

    The ONNX weights are cached outside the repository. This keeps large
    third-party binary files out of Git and avoids repeat downloads.
    """
    if model_path:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"DDColor model file does not exist: {path}")
        return path

    cache_dir = Path.home() / ".cache" / "tintoryx" / "ddcolor"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        downloaded = hf_hub_download(
            repo_id=DDCOLOR_REPO_ID,
            filename=DDCOLOR_FILENAME,
            revision=DDCOLOR_REVISION,
            cache_dir=str(cache_dir),
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not download the pretrained DDColor model. Check your "
            "internet connection, or download the ONNX file manually and "
            "pass --model-path / set TINTORYX_DDCOLOR_MODEL."
        ) from exc

    path = Path(downloaded)
    actual_digest = _sha256(path)
    if actual_digest != DDCOLOR_SHA256:
        raise RuntimeError(
            "The downloaded DDColor model failed SHA-256 verification. "
            "The file may be incomplete or has changed upstream. "
            f"Expected {DDCOLOR_SHA256}, received {actual_digest}."
        )
    return path


def prepare_ddcolor_input(
    image: Image.Image,
    input_size: int = DDCOLOR_INPUT_SIZE,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a normalized NCHW grayscale RGB tensor and retain source Lab L.

    Returns:
        model_input: float32 tensor with shape (1, 3, input_size, input_size).
        luminance: original-resolution Lab L plane, shape (height, width).
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance.")
    if input_size < 1 or max_image_pixels < 1:
        raise ValueError("input_size and max_image_pixels must be positive.")

    oriented = ImageOps.exif_transpose(image).convert("RGB")
    width, height = oriented.size
    if width < 1 or height < 1:
        raise ValueError("Input image has invalid dimensions.")
    if width * height > max_image_pixels:
        raise ValueError(
            f"Image has {width * height:,} pixels; the safety limit is "
            f"{max_image_pixels:,}. Resize the image before uploading."
        )

    rgb = np.asarray(oriented, dtype=np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0].copy()
    del rgb, lab

    # DDColor expects an RGB grayscale conditioning image created by setting
    # Lab a/b to zero, converting Lab back to RGB, and scaling to [0, 1].
    gray_lab = np.zeros((height, width, 3), dtype=np.float32)
    gray_lab[:, :, 0] = luminance
    gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)
    del gray_lab

    resized = cv2.resize(
        gray_rgb,
        (input_size, input_size),
        interpolation=cv2.INTER_CUBIC,
    )
    del gray_rgb
    resized = np.clip(resized, 0.0, 1.0)

    model_input = np.transpose(resized, (2, 0, 1))[None, ...].astype(
        np.float32, copy=False
    )
    return np.ascontiguousarray(model_input), luminance


def reconstruct_rgb(
    luminance: np.ndarray,
    predicted_ab: np.ndarray,
) -> Image.Image:
    """Combine full-resolution Lab luminance with model-predicted chroma."""
    if luminance.ndim != 2:
        raise ValueError("luminance must be a 2-D array.")
    if predicted_ab.ndim != 3 or predicted_ab.shape[2] != 2:
        raise ValueError("predicted_ab must have shape (height, width, 2).")

    height, width = luminance.shape
    ab_full = cv2.resize(
        predicted_ab.astype(np.float32, copy=False),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    lab_result = np.empty((height, width, 3), dtype=np.float32)
    lab_result[:, :, 0] = luminance
    lab_result[:, :, 1:] = ab_full
    rgb_result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2RGB)
    rgb_result = np.clip(rgb_result, 0.0, 1.0)
    return Image.fromarray(
        np.rint(rgb_result * 255.0).astype(np.uint8),
        mode="RGB",
    )


class DDColorColorizer:
    """Run the pinned DDColor-tiny ONNX artifact using ONNX Runtime CPU."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        session: Any | None = None,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    ) -> None:
        if max_image_pixels < 1:
            raise ValueError("max_image_pixels must be a positive integer.")
        self.max_image_pixels = max_image_pixels

        if session is None:
            resolved_path = resolve_model_path(
                model_path or os.environ.get("TINTORYX_DDCOLOR_MODEL")
            )
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError(
                    "onnxruntime is required for DDColor inference. "
                    "Install dependencies with: pip install -r requirements.txt"
                ) from exc

            options = ort.SessionOptions()
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            session = ort.InferenceSession(
                str(resolved_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

        self.session = session
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError("The selected ONNX model has no input or output.")

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        input_shape = inputs[0].shape
        expected_input = [1, 3, DDCOLOR_INPUT_SIZE, DDCOLOR_INPUT_SIZE]
        if input_shape and all(isinstance(value, int) for value in input_shape):
            if input_shape != expected_input:
                raise RuntimeError(
                    f"Unexpected DDColor input shape {input_shape}; "
                    f"expected {expected_input}."
                )

    def colorize(self, image: Image.Image) -> Image.Image:
        """Colorize an image and return an RGB image at original dimensions."""
        model_input, luminance = prepare_ddcolor_input(
            image,
            input_size=DDCOLOR_INPUT_SIZE,
            max_image_pixels=self.max_image_pixels,
        )

        try:
            predictions = self.session.run(
                [self.output_name],
                {self.input_name: model_input},
            )
        except Exception as exc:
            raise RuntimeError(f"DDColor ONNX inference failed: {exc}") from exc

        if not predictions:
            raise RuntimeError("DDColor returned no prediction tensor.")

        predicted = np.asarray(predictions[0], dtype=np.float32)
        expected_output = (1, 2, DDCOLOR_INPUT_SIZE, DDCOLOR_INPUT_SIZE)
        if predicted.shape != expected_output:
            raise RuntimeError(
                f"Unexpected DDColor output shape {predicted.shape}; "
                f"expected {expected_output}."
            )
        if not np.isfinite(predicted).all():
            raise RuntimeError("DDColor returned NaN or infinite values.")

        predicted_ab = np.transpose(predicted[0], (1, 2, 0))
        return reconstruct_rgb(luminance, predicted_ab)
