import unittest
import os, csv, json
import matplotlib.image as mpimg
import numpy as np
from numpy.testing import assert_array_equal
from skimage.measure import compare_ssim as ssim

from src.ea import evolutionary_algorithm
from src.ea.chromosome import Chromosome


class TestEA(unittest.TestCase):

    def setUp(self):
        rel_path = os.path.dirname(__file__)
        image_path = os.path.join(rel_path, "test_images/00000.png")
        self.img_class_16 = mpimg.imread(image_path)
        self.img_class_16 = self.img_class_16[:, :, :3]

        image_path = os.path.join(rel_path, "test_images/00000_gray.png")
        self.img_class_16_gray = mpimg.imread(image_path)
        self.img_class_16_gray = self.img_class_16_gray[:, :, :1]

        image_path =  os.path.join(rel_path, "test_images/00001.png")
        self.img_class_01 = mpimg.imread(image_path)
        self.img_class_01 = self.img_class_01[:, :, :3]

        model = [os.path.join(rel_path, "../res/models/alex_e5_rgb/model.h5")]

        with open(os.path.join(rel_path, "../res/index_label_dict.csv"), "r", encoding="utf8") as f:
            reader = csv.reader(f)
            index_to_label = {rows[0]: rows[1] for rows in reader}

        with open(os.path.join(rel_path, "../config.json"), "r", encoding="utf8") as f:
            data = json.load(f)
            ea_params = data["ea_params_other"]


        self.ea = evolutionary_algorithm.EvolutionaryAlgorithm(0, model, index_to_label,
                                                               ea_params=ea_params,
                                                               color_range=3)
        self.ea._max_gen = 5
        self.ea_gray = evolutionary_algorithm.EvolutionaryAlgorithm(0, model, index_to_label,
                                                                    ea_params=ea_params,
                                                                    color_range=1)
        self.ea_gray._max_gen = 5

    def test_fitness(self):
        self.ea.class_index = 16
        fitness = self.ea.fitness(self.img_class_16)
        self.assertGreater(fitness, 0.9)

        self.ea.class_index = 17
        fitness = self.ea.fitness(self.img_class_16)
        self.assertLess(fitness, 0.001)

        self.ea.class_index = 16
        self.ea._original = self.img_class_16
        fitness = self.ea.fitness(self.img_class_16)
        self.assertLess(fitness, 1.0 / 1000)

    def test_mutate_rgb(self):
        self.ea.class_index = 16
        chrom = Chromosome(self.img_class_16)
        mutated_chrom = self.ea._mutate(0, chrom)
        img_mutated_16 = mutated_chrom.data
        image_diff = ssim(img_mutated_16, self.img_class_16,
                          multichannel=True, data_range=1.0)
        self.assertGreater(image_diff, 0.5)
        self.assertLess(image_diff, 1.0)

    def test_mutate_gray(self):
        self.ea_gray.class_index = 16
        chrom = Chromosome(self.img_class_16_gray)
        mutated_chrom = self.ea_gray._mutate(0, chrom)
        img_mutated_16 = mutated_chrom.data
        image_diff = ssim(np.squeeze(img_mutated_16), np.squeeze(self.img_class_16_gray),
                          multichannel=False, data_range=1.0)
        self.assertGreater(image_diff, 0.5)
        self.assertLess(image_diff, 1.0)


    def test_evaluate_api(self):
        self.ea.class_index = 16
        champion = Chromosome(self.img_class_16)
        api_champ = Chromosome(self.img_class_01)

        # case 1:
        # new champion is better than old champion
        api_champ.api_fitness = -1
        champion.fitness = 0.9
        new_champ, reset = self.ea._evaluate_api(champion, api_champ)
        self.assertFalse(reset)
        assert_array_equal(new_champ.data, self.img_class_16)

        # case 2:
        # new champion is only slightly worse than old champion
        api_champ.api_fitness = 0.999999999
        new_champ, reset = self.ea._evaluate_api(champion, api_champ)
        self.assertIsNone(new_champ)
        self.assertFalse(reset)

        # case 3:
        # new champion is way worse than api champion
        api_champ.api_fitness = 0.999999999
        champion.data = self.img_class_01
        api_champ.data = self.img_class_16
        new_champ, reset = self.ea._evaluate_api(champion, api_champ)
        self.assertIsNone(new_champ)
        self.assertTrue(reset)
