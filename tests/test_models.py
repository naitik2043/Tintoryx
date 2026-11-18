import unittest

import torch

from src.metrics import batch_psnr
from src.models import PatchDiscriminator, UNetGenerator


class TintoryxModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_unet_maps_grayscale_to_rgb(self):
        model = UNetGenerator(base=8).eval()
        grayscale = torch.rand(2, 1, 32, 32)

        with torch.no_grad():
            prediction = model(grayscale)

        self.assertEqual(tuple(prediction.shape), (2, 3, 32, 32))
        self.assertTrue(torch.all(prediction >= 0).item())
        self.assertTrue(torch.all(prediction <= 1).item())

    def test_patchgan_returns_patch_logits(self):
        model = PatchDiscriminator(base=8).eval()
        grayscale = torch.rand(2, 1, 32, 32)
        rgb = torch.rand(2, 3, 32, 32)

        with torch.no_grad():
            logits = model(grayscale, rgb)

        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[1], 1)
        self.assertGreater(logits.shape[2], 1)
        self.assertGreater(logits.shape[3], 1)

    def test_perfect_prediction_has_maximum_clamped_psnr(self):
        image = torch.rand(2, 3, 32, 32)
        score = batch_psnr(image, image)

        self.assertAlmostEqual(score, 100.0, places=4)

    def test_larger_error_has_lower_psnr(self):
        target = torch.zeros(2, 3, 32, 32)
        small_error = torch.full_like(target, 0.1)
        large_error = torch.full_like(target, 0.5)

        self.assertGreater(
            batch_psnr(small_error, target),
            batch_psnr(large_error, target),
        )


if __name__ == "__main__":
    unittest.main()
