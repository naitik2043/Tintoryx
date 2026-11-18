# Tintoryx - MSE2 Examination Project Notes

## Project identity

- **Name:** Tintoryx
- **Full title:** Tintoryx: Paired Conditional GAN for CIFAR-10 Image Colorization
- **Course reference:** MSE2-DLT-050 - Historical Photo Restoration (Paired Conditional GAN)
- **Implementation scope:** paired grayscale-to-RGB colorization on 32 x 32 CIFAR-10 images

## Assignment requirements

| Item | Specification |
|---|---|
| Task | Colorization of grayscale images using paired conditional GAN |
| Dataset | CIFAR-10 |
| Training subset | 10,000 images |
| Test subset | 1,000 images |
| Input | 1-channel grayscale, 32 x 32 |
| Target/output | 3-channel RGB, 32 x 32 |
| Generator | Lightweight U-Net |
| Discriminator | Conditional PatchGAN |
| CGAN generator loss | Adversarial loss + 10 x L1 loss |
| Baseline | U-Net with L1 loss only |
| Training | 12 epochs per model |
| Evaluation | Average test PSNR |
| Visual result | Grayscale / L1 output / CGAN output / ground-truth RGB |

## Implemented pipeline

1. Load CIFAR-10 RGB images using Torchvision.
2. Select deterministic training and test subsets using seed 42.
3. Convert each RGB image to a paired 1-channel grayscale input.
4. Train the U-Net baseline with L1 reconstruction loss.
5. Train the U-Net generator and conditional PatchGAN discriminator.
6. Evaluate both generators on the same 1,000-image test subset.
7. Save a PSNR comparison chart and visual prediction grids.

## Model objectives

**Baseline**

`L_baseline = L1(prediction, target)`

**Conditional GAN generator**

`L_G = L_adversarial + 10 x L1(prediction, target)`

The discriminator is trained to classify real grayscale/RGB pairs as real and generated grayscale/RGB pairs as fake.

## Training configuration used

| Parameter | Value |
|---|---:|
| Epochs | 12 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 0.0002 |
| Beta 1 | 0.5 |
| L1 weight | 10 |
| Seed | 42 |
| GPU | NVIDIA GeForce RTX 2050 |

Environment used for the reported run: Python 3.12.10, PyTorch 2.14.1+cu130, Torchvision 0.29.1+cu130, CUDA available.

## Measured results

Both trained models were evaluated on the 1,000-image test subset.

| Model | Average test PSNR |
|---|---:|
| L1-only U-Net | **24.201 dB** |
| Conditional GAN | 22.606 dB |
| CGAN - baseline | -1.595 dB |

The L1-only baseline achieved the higher PSNR, indicating lower pixel-wise reconstruction error. The CGAN generated more vivid colors in some examples, but its colors were not always closer to the ground truth. PSNR does not fully capture perceptual quality.

## Output artifacts

- `outputs/dataset_pair_preview.png` - paired grayscale and RGB samples.
- `outputs/tintoryx_restoration_strip.png` - four-column model comparison.
- `outputs/tintoryx_trained_predictions.png` - CGAN inference samples.
- `outputs/psnr_comparison.png` - measured average test PSNR comparison.

## Recommended presentation order

1. Problem and motivation
2. Dataset and paired preprocessing
3. U-Net generator and PatchGAN discriminator
4. Training objectives: baseline versus CGAN
5. Experimental setup
6. Four-column visual comparison
7. PSNR comparison and interpretation
8. Limitations and future work

## Viva talking points

1. **Why paired data?** Each grayscale input has its corresponding RGB target, enabling supervised reconstruction.
2. **Why U-Net?** Its encoder-decoder structure learns features while skip connections preserve spatial information.
3. **Why a discriminator?** It encourages generated local color and texture patterns to resemble real image pairs.
4. **Why L1?** It penalizes absolute pixel differences and anchors predictions to their paired target.
5. **Why combine adversarial and L1 losses?** Adversarial loss encourages realism; L1 encourages fidelity to the target.
6. **Why use PSNR and visual comparisons?** PSNR provides a numeric pixel-error measure, while visual inspection reveals color differences and artifacts.

## Limitations and scope clarification

The assignment heading mentions historical photo restoration, but the specified implementation dataset is CIFAR-10 at 32 x 32 resolution. This project is therefore a small-image colorization demonstration, not a validated high-resolution archival-photo restoration system. A higher-resolution historical-photo dataset and additional restoration methods would be required to support that broader claim.

## Future work

- Larger paired datasets and higher-resolution generators.
- Perceptual losses and additional image-quality metrics.
- Per-epoch training history and learning curves.
- A user-facing interface for arbitrary grayscale image inference.
