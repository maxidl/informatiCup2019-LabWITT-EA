import unittest
import os

from src.utils.image_utils import load_image

class TestImageUtils(unittest.TestCase):


    def test_image_input_shape(self):
        rel_path = os.path.dirname(__file__)

        img_0 = load_image(os.path.join(rel_path, "test_images/0_true_grayscale.png"))
        img_1 = load_image(os.path.join(rel_path, "test_images/1_true_rgb.png"))
        img_2 = load_image(os.path.join(rel_path, "test_images/2_grayscale_normed.png"))
        img_3 = load_image(os.path.join(rel_path, "test_images/3_rgba.png"))
        img_4 = load_image(os.path.join(rel_path, "test_images/4_rgba_125x72.png"), 64)

        self.assertEqual(img_0.shape, (64, 64))
        self.assertEqual(img_1.shape, (64, 64, 3))
        self.assertEqual(img_2.shape, (64, 64))
        self.assertEqual(img_3.shape, (64, 64, 3))
        self.assertEqual(img_4.shape, (64, 64, 3))


if __name__ == '__main__':
    unittest.main()


