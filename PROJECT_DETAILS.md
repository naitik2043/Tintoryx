# Tintoryx — MSE2 Examination Project Notes

## Project identity
**Name:** Tintoryx  
**Full title:** Tintoryx: Paired Conditional GAN for CIFAR-10 Image Colorization  
**Course reference:** MSE2-DLT-050 — Historical Photo Restoration (Paired Conditional GAN)

## Exact requirements interpreted from the brief
| Item | Required specification |
|---|---|
| Task | Automated colorization of grayscale images using paired CGAN |
| Dataset | CIFAR-10 |
| Training subset | 10,000 images |
| Test subset | 1,000 images |
| Input | 1-channel grayscale, 32×32 |
| Target/output | 3-channel RGB, 32×32 |
| Generator | Lightweight U-Net |
| CGAN loss | Adversarial loss + 10 × L1 loss |
| Baseline | U-Net with L1 loss only; no discriminator |
| Training | 12 epochs |
| Visual result | Four-column strip: Grayscale Input / L1-only Output / CGAN Colorized / Ground Truth RGB |
| Numeric result | Bar chart comparing average test PSNR (dB) |

## Recommended presentation order
1. Problem and motivation
2. Dataset and paired preprocessing
3. Proposed architecture
4. Training objectives: baseline vs CGAN
5. Experimental setup
6. Four-column restoration strip
7. PSNR comparison
8. Limitations and conclusion

## Suggested conclusion template
“Tintoryx compares a supervised U-Net baseline with a conditional GAN for paired CIFAR-10 colorization. The L1-only model focuses on pixel-level reconstruction, while the CGAN combines adversarial learning with a weighted L1 term. The final conclusion should be based on the measured test PSNR and the generated visual comparison from the actual experiment.”

## Important honesty point
The assignment's heading mentions historical photo restoration, but its specified dataset is CIFAR-10 (32×32). Therefore, this implementation is a small-image colorization demonstration and not a validated system for restoring real historical photographs.
