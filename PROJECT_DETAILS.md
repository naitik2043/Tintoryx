# Tintoryx — Project and Experiment Notes

## Project identity

- **Name:** Tintoryx
- **Academic title:** Tintoryx: Paired Conditional GAN for CIFAR-10 Image Colorization
- **Course reference:** MSE2-DLT-050 — Historical Photo Restoration (Paired Conditional GAN)
- **Implemented coursework scope:** paired grayscale-to-RGB colorization on 32 × 32 CIFAR-10 images
- **Recommended photograph inference:** pretrained DDColor-tiny ONNX backend

## 1. Two project tracks

Tintoryx intentionally keeps two separate model paths.

| Track | Model | Intended role |
|---|---|---|
| Photograph colorization | Pretrained DDColor-tiny ONNX, 512 × 512 | Recommended upload and CLI experience |
| Academic experiment | U-Net baseline and Conditional GAN, 32 × 32 CIFAR-10 | Coursework, training, evaluation, and model comparison |

The original U-Net/cGAN was trained on small CIFAR-10 images and is not suitable for general photograph colorization. The DDColor model is a separate pretrained model, not a cGAN checkpoint trained by this repository.

## 2. Why the original output looked blurry

The first photo upload path reduced an arbitrary photograph to 32 × 32 before passing it to a U-Net trained on CIFAR-10. It then enlarged the 32 × 32 RGB prediction. The resize step could not reconstruct detail lost during downsampling.

The recommended path now predicts Lab chroma at 512 × 512 and keeps the source photograph's original-resolution Lab luminance channel. It resizes predicted chroma back to the input size, combines the Lab channels, and converts the result to RGB.

Output dimensions match the source image, but the chroma prediction is still limited to 512 × 512. It is not native-resolution color generation or a complete photo-restoration pipeline.

## 3. DDColor-tiny model

- **Original method:** DDColor, ICCV 2023.
- **Original implementation:** https://github.com/piddnad/DDColor
- **ONNX artifact:** https://huggingface.co/toolsmonk/ddcolor-tiny
- **Pinned model repository revision:** c7d348d
- **Model file:** `ddcolor_tiny_512_int8.onnx`
- **Model size:** 112,723,138 bytes (~107.5 MiB)
- **Input:** float32 `[1, 3, 512, 512]`, RGB values in [0, 1].
- **Output:** float32 `[1, 2, 512, 512]`, Lab a/b planes.
- **Artifact SHA-256:** `e983347272b8e218edc974a0af2bd66f53b73212abb800e103c64806e4c14a14`
- **Model license:** Apache License 2.0, as reported by the model card.
- **Runtime:** ONNX Runtime CPU execution provider.
- **Model storage:** downloaded on first use, cached outside the Git repository.

The Hugging Face model card describes the artifact as a dynamic-quantized DDColor-tiny export with MatMul and Gemm weights quantized to uint8. It reports a mean absolute difference of approximately 1.1 Lab chroma units from its FP32 export on a test image. Tintoryx verifies the downloaded binary against the published checksum.

### DDColor inference sequence

1. Apply EXIF orientation and convert the source to RGB.
2. Convert the image to floating-point CIE Lab.
3. Retain the original-resolution Lab L plane.
4. Set Lab a and b to zero and convert the grayscale Lab image back to RGB.
5. Resize the grayscale conditioning image to 512 × 512 and scale it to [0, 1].
6. Run the ONNX model to predict the two Lab chroma planes.
7. Resize the predicted a/b planes to the original source dimensions.
8. Combine source L and predicted a/b planes.
9. Convert Lab to RGB and save the colorized result.

## 4. DDColor application usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Run command-line inference:

```bash
python -m src.infer --input photo.jpg --output outputs/photo_colorized.png
```

Override the downloaded model path:

```powershell
$env:TINTORYX_DDCOLOR_MODEL="models/ddcolor_tiny_512_int8.onnx"
python -m src.infer --input photo.jpg
```

The first model download requires internet access. Later inference reuses the local Hugging Face cache.

## 5. Academic assignment specification

| Item | Specification |
|---|---|
| Task | Grayscale-to-RGB image colorization |
| Dataset | CIFAR-10 |
| Training subset | 10,000 images |
| Test subset | 1,000 images |
| Input | 1-channel grayscale, 32 × 32 |
| Target/output | 3-channel RGB, 32 × 32 |
| Generator | Lightweight U-Net |
| Discriminator | Conditional PatchGAN |
| Baseline objective | L1 reconstruction loss |
| cGAN generator objective | Adversarial loss + 10 × L1 loss |
| Training | 12 epochs per model |
| Evaluation | PSNR and SSIM |
| Visualization | Grayscale / baseline / cGAN / original RGB |

## 6. Academic training configuration

| Parameter | Value |
|---|---:|
| Epochs | 12 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 0.0002 |
| Beta 1 | 0.5 |
| L1 weight | 10 |
| Seed | 42 |
| Image size | 32 × 32 |
| GPU in documented run | NVIDIA GeForce RTX 2050 |

The training script seeds Python, NumPy, PyTorch, CUDA, and DataLoader shuffling. Exact determinism can still vary with hardware and numerical libraries.

## 7. Previously recorded academic PSNR results

| Model | Average test PSNR |
|---|---:|
| L1-only U-Net | **24.201 dB** |
| Conditional GAN | 22.606 dB |
| CGAN − baseline | **−1.595 dB** |

The L1-only U-Net achieved higher PSNR on the reported test subset, indicating lower pixel-wise reconstruction error. The cGAN generated more vivid colors in some examples, but not always colors closer to the ground truth.

These results belong only to the original CIFAR-10 experiment. They are not DDColor scores.

## 8. Academic evaluation metrics

### PSNR

```text
MSE  = mean((prediction - target) ** 2)
PSNR = 10 × log10(1 / MSE)
```

PSNR is calculated per image for pixel values in [0, 1], then averaged over the test subset.

### SSIM

SSIM compares local luminance, contrast, and structure. The implementation uses a Gaussian window and a data range of 1.0. The evaluation script writes the computed PSNR and SSIM values to `outputs/evaluation_results.json` and creates a comparison chart.

Do not report new SSIM values until the updated evaluation is run on the locally trained model checkpoints.

## 9. Important source files

| File | Responsibility |
|---|---|
| `app.py` | Streamlit interface and backend selector |
| `src/ddcolor.py` | Download verification, preprocessing, ONNX inference, Lab reconstruction |
| `src/infer.py` | CLI inference for DDColor or the academic cGAN |
| `src/data.py` | CIFAR-10 loading and paired grayscale conversion |
| `src/models.py` | U-Net generator and PatchGAN discriminator |
| `src/train.py` | Baseline and cGAN training loops |
| `src/training_artifacts.py` | Training-history JSON and loss plots |
| `src/metrics.py` | PSNR and SSIM |
| `src/evaluate.py` | Held-out evaluation and charts |
| `tests/test_ddcolor.py` | DDColor pipeline tests with a mock ONNX session |
| `tests/test_models.py` | Academic architecture and metric tests |
| `docs/THIRD_PARTY_NOTICES.md` | DDColor provenance, attribution, checksum, and license |

## 10. Limitations

- DDColor predicts plausible colors; original colors cannot be known exactly from grayscale alone.
- Chroma is predicted at 512 × 512 and resized to source dimensions; native-resolution chroma is not generated.
- DDColor is not a scratch removal, denoising, deblurring, or face-restoration model.
- The academic U-Net/cGAN is restricted to 32 × 32 CIFAR-10 images.
- Application upload limit is 20 MB and default image pixel limit is 16 megapixels.
- First-use model retrieval requires an internet connection.
- Quality depends on image content, contrast, and condition.

## 11. Future work

- Evaluate DDColor on a curated paired photo benchmark with PSNR, SSIM, LPIPS, and visual review.
- Explore tiled or multi-scale inference for larger photographs.
- Fine-tune on a licensed high-resolution paired dataset.
- Integrate dedicated restoration models for scratch removal, denoising, and deblurring.
- Add performance profiling and hardware-specific acceleration.

## 12. Reproduce the academic experiment

```bash
python -m src.prepare_data
python -m src.train --model both
python -m src.evaluate
python -m src.visualize
python -m unittest discover -s tests -v
```

For photograph colorization, run `streamlit run app.py` or use the DDColor command in `src/infer.py`.
