import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.ddcolor import DDColorColorizer, prepare_ddcolor_input


class FakeDDColorSession:
    def __init__(self, chroma=(0.0, 0.0)):
        self._inputs = [
            SimpleNamespace(name="input", shape=[1, 3, 512, 512])
        ]
        self._outputs = [
            SimpleNamespace(name="output", shape=[1, 2, 512, 512])
        ]
        self.chroma = chroma
        self.last_input = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, inputs):
        self.last_input = inputs["input"]
        output = np.zeros((1, 2, 512, 512), dtype=np.float32)
        output[:, 0, :, :] = self.chroma[0]
        output[:, 1, :, :] = self.chroma[1]
        return [output]


class DDColorPipelineTests(unittest.TestCase):
    def test_preprocess_creates_expected_model_input(self):
        image = Image.new("RGB", (1280, 720), color=(90, 120, 150))
        model_input, luminance = prepare_ddcolor_input(image)

        self.assertEqual(model_input.shape, (1, 3, 512, 512))
        self.assertEqual(model_input.dtype, np.float32)
        self.assertTrue(np.isfinite(model_input).all())
        self.assertGreaterEqual(float(model_input.min()), 0.0)
        self.assertLessEqual(float(model_input.max()), 1.0)
        self.assertEqual(luminance.shape, (720, 1280))

    def test_colorizer_preserves_original_dimensions(self):
        session = FakeDDColorSession()
        colorizer = DDColorColorizer(session=session)
        image = Image.new("L", (1365, 768), color=128)

        result = colorizer.colorize(image)

        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, image.size)
        self.assertEqual(session.last_input.shape, (1, 3, 512, 512))

    def test_neutral_chroma_keeps_grayscale_neutral(self):
        colorizer = DDColorColorizer(session=FakeDDColorSession())
        image = Image.new("L", (320, 200), color=100)

        result = np.asarray(colorizer.colorize(image), dtype=np.int16)
        red_green = np.abs(result[:, :, 0] - result[:, :, 1]).max()
        green_blue = np.abs(result[:, :, 1] - result[:, :, 2]).max()

        self.assertLessEqual(int(red_green), 1)
        self.assertLessEqual(int(green_blue), 1)

    def test_nonzero_chroma_produces_color(self):
        colorizer = DDColorColorizer(
            session=FakeDDColorSession(chroma=(35.0, -25.0))
        )
        image = Image.new("L", (160, 120), color=140)

        result = np.asarray(colorizer.colorize(image), dtype=np.int16)
        channel_spread = (
            result.max(axis=2).astype(np.int16)
            - result.min(axis=2).astype(np.int16)
        )
        self.assertGreater(int(channel_spread.max()), 5)

    def test_input_pixel_safety_limit(self):
        colorizer = DDColorColorizer(
            session=FakeDDColorSession(),
            max_image_pixels=100,
        )
        image = Image.new("RGB", (11, 10), color="white")

        with self.assertRaises(ValueError):
            colorizer.colorize(image)

    def test_rejects_unexpected_model_output_shape(self):
        class InvalidSession(FakeDDColorSession):
            def run(self, output_names, inputs):
                return [np.zeros((1, 3, 512, 512), dtype=np.float32)]

        colorizer = DDColorColorizer(session=InvalidSession())
        with self.assertRaises(RuntimeError):
            colorizer.colorize(Image.new("L", (64, 64), color=128))


if __name__ == "__main__":
    unittest.main()
