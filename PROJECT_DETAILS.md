# Tintoryx — Project and Experiment Notes

## Project identity

- **Name:** Tintoryx
- **Full title:** Tintoryx: Paired Conditional GAN for CIFAR-10 Image Colorization
- **Course reference:** MSE2-DLT-050 — Historical Photo Restoration (Paired Conditional GAN)
- **Implemented scope:** paired grayscale-to-RGB colorization on 32 × 32 CIFAR-10 images

## 1. Problem statement

A grayscale image contains intensity information but does not explicitly contain the original colors. Tintoryx learns a mapping from grayscale images to plausible RGB predictions using paired examples.

The task is ambiguous: multiple color images can correspond to a similar grayscale image. The model therefore learns color patterns from the training dataset rather than recovering uniquely known colors.

**Scope clarification:** although the assignment heading mentions historical-photo restoration, the actual implementation uses CIFAR-10 at 32 × 32 pixels. The project demonstrates small-image colorization and is not a validated high-resolution photo-restoration tool.

## 2. Assignment and implementation specification

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

## 3. Implemented pipeline

1. Load CIFAR-10 RGB samples using Torchvision.
2. Select deterministic train and test subsets using seed 42.
3. Convert each RGB sample into a paired one-channel grayscale input.
4. Train a U-Net baseline using L1 reconstruction loss.
5. Train a U-Net generator and conditional PatchGAN discriminator.
6. Record epoch-level training losses and save a training-history plot.
7. Evaluate both generators on the same held-out test subset.
8. Calculate average per-image PSNR and SSIM.
9. Save metric results as JSON and comparison charts.
10. Provide command-line and Streamlit inference on uploaded images, with the current 32 × 32 model input limitation.

## 4. Model objectives

### U-Net baseline

```text
L_baseline = L1(prediction, target)
```

L1 loss is the mean absolute difference between predicted and target pixel values. It encourages the prediction to remain close to the paired RGB target.

### Conditional GAN

```text
L_G = L_adversarial + 10 × L1(prediction, target)
```

- The generator tries to create a convincing colorized image while staying close to the paired target.
- The discriminator receives a grayscale image concatenated with either a real RGB target or generated RGB output.
- The PatchGAN discriminator predicts local real/fake logits.
- The discriminator uses binary cross-entropy with logits.

## 5. Training configuration

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

## 6. Previously recorded PSNR results

The repository's documented experiment reports:

| Model | Average test PSNR |
|---|---:|
| L1-only U-Net | **24.201 dB** |
| Conditional GAN | 22.606 dB |
| CGAN − baseline | **−1.595 dB** |

The L1-only U-Net achieved the higher PSNR on this test subset, indicating lower pixel-wise reconstruction error. The cGAN produced more vivid colors in some examples, but its predictions were not always closer to the ground-truth pixels.

### Metric additions

The updated evaluation pipeline also calculates SSIM. Updated SSIM values are generated when the training and evaluation scripts are run with the local checkpoints. They are written to `outputs/evaluation_results.json`; no unmeasured SSIM result is reported here.

## 7. Evaluation metric definitions

### PSNR

```text
MSE  = mean((prediction - target) ** 2)
PSNR = 10 × log10(1 / MSE)
```

PSNR is calculated per image for pixel values in [0, 1], then averaged over the test subset. Higher PSNR indicates lower pixel-wise reconstruction error. The implementation clamps zero MSE to a small positive value, so identical images return a finite 100 dB.

### SSIM

SSIM compares local luminance, contrast, and structure. Tintoryx uses a Gaussian window and a data range of 1.0. The implementation calculates an SSIM score for each image and averages the scores across the batch and test subset.

SSIM complements PSNR, but neither metric completely describes perceptual color realism.

## 8. Output artifacts

| Artifact | Description |
|---|---|
| `outputs/dataset_pair_preview.png` | Paired grayscale and RGB samples |
| `outputs/tintoryx_restoration_strip.png` | Four-column model comparison |
| `outputs/tintoryx_trained_predictions.png` | cGAN predictions |
| `outputs/training_history.json` | Epoch-level training losses and config |
| `outputs/training_history.png` | Baseline and cGAN loss curves |
| `outputs/evaluation_results.json` | PSNR, SSIM, sample counts, metric deltas |
| `outputs/metrics_comparison.png` | PSNR and SSIM comparison chart |

Generated histories and evaluation outputs are local experiment artifacts and are excluded from Git tracking.

## 9. Important source files

| File | Responsibility |
|---|---|
| `config.py` | Training configuration and compute device |
| `src/data.py` | CIFAR-10 loading and paired grayscale conversion |
| `src/models.py` | U-Net generator and PatchGAN discriminator |
| `src/train.py` | Baseline and cGAN training loops |
| `src/training_artifacts.py` | Training-history JSON and loss plot |
| `src/metrics.py` | PSNR and SSIM calculations |
| `src/evaluate.py` | Held-out metric evaluation and charts |
| `src/visualize.py` | Side-by-side model output visualization |
| `src/infer.py` | Command-line image inference |
| `app.py` | Streamlit upload and colorization interface |
| `tests/test_models.py` | Architecture and metric unit tests |

## 10. Viva talking points

1. **Why paired data?** Each grayscale image is generated from a known RGB image, so the model can learn from its matching target.
2. **Why U-Net?** The encoder-decoder learns image features, while skip connections help retain spatial information.
3. **Why L1 loss?** It penalizes absolute pixel differences and anchors the output to the paired RGB target.
4. **Why a discriminator?** It encourages generated image patches to resemble patches from real RGB targets.
5. **Why Conditional GAN?** The discriminator receives the grayscale condition along with the candidate RGB image, so it judges whether the colorized result is plausible for that input.
6. **Why use both PSNR and SSIM?** PSNR measures pixel error; SSIM compares local structural similarity. They provide complementary evidence.
7. **Why did the baseline achieve higher PSNR?** The adversarial objective can favor realistic-looking local patterns rather than exact pixel matching. In this recorded run, that trade-off resulted in lower PSNR for the cGAN.
8. **Does the app restore high-resolution photographs?** No. The current model was trained at 32 × 32 on CIFAR-10. The interface resizes uploaded images for a demonstration only.

## 11. Limitations

- CIFAR-10 images are small and contain limited object categories.
- Grayscale-to-color prediction is inherently ambiguous.
- The training subset and epoch count are limited.
- PSNR and SSIM do not fully capture perceptual color quality.
- Arbitrary-image inference downsamples the input to 32 × 32, then resizes the generated result to the original dimensions.
- The model is not a production-ready historical-photo restoration system.

## 12. Future work

- Train on larger paired datasets and higher-resolution images.
- Add perceptual and color-consistency losses.
- Add per-epoch validation and automatic best-checkpoint selection.
- Compare multiple random seeds and report mean and standard deviation.
- Evaluate with additional perceptual metrics and human assessment.
- Train a dedicated high-resolution model before claiming archival-restoration capability.

## 13. Reproduction commands

```bash
python -m src.prepare_data
python -m src.train --model both
python -m src.evaluate
python -m src.visualize
python -m unittest discover -s tests -v
streamlit run app.py
```
