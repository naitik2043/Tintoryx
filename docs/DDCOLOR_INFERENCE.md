# DDColor inference backend

Tintoryx now separates its educational CIFAR-10 experiment from the default
photograph-colorization experience.

## Why the previous output looked blurry

The original U-Net/cGAN experiment was trained on CIFAR-10 images at
32 × 32 pixels. The initial upload interface reduced every photograph to
32 × 32, ran the model, and enlarged the generated result. Upscaling cannot
restore detail removed before inference.

The CIFAR-10 models remain in the repository for coursework and controlled
baseline-vs-cGAN comparisons. They are no longer the recommended engine for
photograph colorization.

## Recommended backend: DDColor-tiny

The web application and command-line tool use a pretrained, quantized
DDColor-tiny ONNX model. DDColor is an ICCV 2023 image-colorization method.
The selected ONNX artifact is designed for a 512 × 512 input and returns two
Lab chroma planes.

On first use, Tintoryx downloads the 107.5 MiB ONNX file from
`toolsmonk/ddcolor-tiny` at pinned revision `c7d348d`. The downloaded
artifact is checked against the SHA-256 in
[`docs/THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before inference.
The file is cached outside the Git repository.

The model predicts Lab a/b chroma. Tintoryx retains the original-resolution
Lab L channel, creates a grayscale RGB conditioning image by setting a=b=0,
runs chroma prediction at 512 × 512, resizes predicted a/b planes to the
original image dimensions, then combines them with the source L channel.
This keeps the original luminance detail and avoids shrinking the entire
colorized output to 32 × 32.

The chroma prediction itself is still made at 512 × 512. Keeping the source
dimensions does not mean that the model produces native-resolution chroma,
repairs scratches, removes blur, or restores lost image detail. Generated
colors are plausible predictions and cannot guarantee historical accuracy.

## Setup

```bash
pip install -r requirements.txt
```

The default ONNX model is quantized and runs through ONNX Runtime's CPU
execution provider. The first model download requires an internet connection;
later runs use the local cache. Users can also download the file manually and
set `TINTORYX_DDCOLOR_MODEL` to its path.

## Streamlit interface

```bash
streamlit run app.py
```

Choose **DDColor — pretrained photo colorizer (recommended)**, upload an image,
and download the generated PNG. The first run downloads the model once.

The **Tintoryx U-Net — CIFAR-10 academic experiment** option remains available
for comparison. It requires
`checkpoints/tintoryx_cgan_generator.pt` and is expected to look blurry on
real photographs.

## Command line

Run the recommended pretrained model:

```bash
python -m src.infer --input photo.jpg --output outputs/photo_colorized.png
```

Use a local copy of the ONNX model:

```bash
python -m src.infer --input photo.jpg --output outputs/photo_colorized.png --model-path models/ddcolor_tiny_512_int8.onnx
```

Or set the environment variable in Windows PowerShell:

```powershell
$env:TINTORYX_DDCOLOR_MODEL="models/ddcolor_tiny_512_int8.onnx"
python -m src.infer --input photo.jpg
```

Run the academic CIFAR-10 cGAN explicitly:

```bash
python -m src.infer --input photo.jpg --model cgan
```

## Runtime constraints

- Maximum image dimension budget: 16 megapixels by default.
- Streamlit upload size: 20 MB.
- Model file: approximately 107.5 MiB; cached outside the repository.
- Inference uses ONNX Runtime's CPU execution provider for broad compatibility.
- Large images require more memory; resize unusually large source files before
  inference if the available RAM is limited.
