import random
import mahotas.polygon
import numpy as np

from src.ea.chromosome import Chromosome


class Poly:
    """ Class for creating polygons

    """
    def __init__(self, width_x, width_y, color_range):
        """

        Args:
            width_x:
            width_y:
            color_range:
        """
        self.width_x = width_x
        self.width_y = width_y
        self._color_range = color_range

    def create_poly(self):
        """Create a two dimensional polygon

        Returns: polygon

        """
        # create a two dimensional polygon
        edges = random.randint(2, 8)
        pts = []
        poly = np.zeros((self.width_x, self.width_y, self._color_range))
        for i in range(edges):
            pts.append((random.randint(-1, self.width_x - 1), random.randint(0, self.width_y - 1)))
        mahotas.polygon.fill_polygon(pts, poly, color=1.0)

        if self._color_range == 3:
            red = random.random()
            green = random.random()
            blue = random.random()
            poly[:, :, 0] = poly[:, :, 0] * red
            poly[:, :, 1] = poly[:, :, 1] * green
            poly[:, :, 2] = poly[:, :, 2] * blue
        else:
            gray = random.random()
            poly = poly * gray

        return poly

    def create_empty_image(self):
        return np.zeros((self.width_x, self.width_y, self._color_range))

    def add_poly(self, img: np.ndarray):
        """Add a polygon to an image

        Args:
            img (np.ndarray): input image

        Returns: image with added polygon (np.ndarray)

        """
        poly = self.create_poly()
        image = np.add(img, poly)
        image[poly.astype(np.bool)] = image[poly.astype(np.bool)] / 2.0
        return image

    def create_random_chromosome(self):
        """Create a random chromosome

        Initalize each pixel randomly. We are not smarter than the EA, we do not give any hints.

        Returns: a new chromosome (Chromosome)

        """
        chrom = Chromosome(np.zeros((self.width_x, self.width_y, self._color_range)))
        chrom.data = self.add_poly(chrom.data)
        return chrom

    def mutate(self, chrom: Chromosome, n: None) -> Chromosome:
        """ Mutate a chromosome

        Args:
            chrom (Chromosome): chromosome
            n (int): not implemented here

        Returns: a new chromosome (Chromosome)

        """
        img = np.copy(chrom.data)
        img = self.add_poly(img)
        return Chromosome(data=img, fitness=-1)
