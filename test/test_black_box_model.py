import unittest
import os
import numpy as np


from src.blackbox.black_box_model import LocalModel

class TestBlackBoxModel(unittest.TestCase):

    def setUp(self):

        model_path = os.path.join(os.path.dirname(__file__), "../res/models/alex_e5_rgb/model.h5")
        self.model = LocalModel(model_path)

        self.img_64_1 = np.random.rand(64, 64, 1)
        self.img_64_3 = np.random.rand(64, 64, 3)

    def test_true_grayscale_image(self):
        self.assertRaises(ValueError, self.model.send_query, np.random.rand(64, 64))


    def test_response(self):

        indexes, probs = self.model.send_query(self.img_64_1)
        self.assertEqual(indexes.size, 43)
        self.assertEqual(probs.size, 43)
        indexes, probs = self.model.send_query(self.img_64_3)
        self.assertEqual(indexes.size, 43)
        self.assertEqual(probs.size, 43)
