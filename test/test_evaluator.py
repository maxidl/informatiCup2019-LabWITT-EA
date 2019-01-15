import unittest
import os, csv
import matplotlib.image as mpimg

from src.ea import evaluator

class TestEvaluator(unittest.TestCase):

    # Load the evaluator
    # Load some test images
    def setUp(self):
        rel_path = os.path.dirname(__file__)
        image_path = os.path.join(rel_path, "test_images/00000.png")
        self.img_class_16 = mpimg.imread(image_path)
        self.img_class_16 = self.img_class_16[:, :, :3]

        image_path =  os.path.join(rel_path, "test_images/00001.png")
        self.img_class_01 = mpimg.imread(image_path)
        self.img_class_01 = self.img_class_01[:, :, :3]

        model = [os.path.join(rel_path, "../res/models/alex_e5_rgb/model.h5")]

        with open(os.path.join(rel_path, "../res/index_label_dict.csv"), "r", encoding="utf8") as \
                csv_file:
            reader = csv.reader(csv_file)
            index_to_label = {rows[0]: rows[1] for rows in reader}

        self.eval_ = evaluator.Evaluator(index_to_label, model)

    # test our own nn
    # should return high confidence for valid class
    # and low confidence for wrong class
    def test_eval_own_nn(self):
        conf = self.eval_.eval_own_nn(self.img_class_16, 16)
        self.assertGreater(conf, 0.90)

        conf = self.eval_.eval_own_nn(self.img_class_16, 17)
        self.assertLess(conf, 0.001)

        conf = self.eval_.eval_own_nn(self.img_class_01, 1)
        self.assertGreater(conf, 0.90)

        conf = self.eval_.eval_own_nn(self.img_class_01, 8)
        self.assertLess(conf, 0.001)

    # test the api call
    # high confidence for valid class
    # -1 for wrong class
    def test_eval(self):
        conf = self.eval_.eval(self.img_class_16, 16)
        self.assertGreater(conf, 0.90)

        conf = self.eval_.eval(self.img_class_16, 17)
        self.assertEqual(conf, -1)
