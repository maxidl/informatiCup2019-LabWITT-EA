import numpy as np

from src.ea.chromosome import Chromosome


class Standard:

    def __init__(self, image_size_x: int, image_size_y: int, color_range: int,
                 mutation_prob: float, mutation_step: float, mutation_linear: bool, max_gen: int):
        """

        Args:
            image_size_x: image input with in pixel
            image_size_y: image input height in pixel
            color_range:  color range, either 1 for grayscale or 3 for rgb
            mutation_prob: probability to mutate
            mutation_step: maximum value to change one pixel
            mutation_linear: dynamic mutation step by linear mutation, else static
            max_gen: number of maximum generations
        """
        self.__width_x = image_size_x
        self.__width_y = image_size_y
        if color_range == 1 or color_range == 3:
            self.__color_range = color_range
        else:
            raise ValueError("Color range have to be 1 or 3!")

        self.__mutation_prob = mutation_prob
        self.__mutation_step = mutation_step
        self.__mutation_linear = mutation_linear
        self.__max_gen = max_gen

    def create_random_chromosome(self) -> Chromosome:
        """Create a random chromosome at pixel space

        Returns: generated chromosome (Chromosome)

        """
        return Chromosome(np.random.rand(self.__width_x, self.__width_y, self.__color_range))

    def mutate(self, chrom: Chromosome, n: float) -> Chromosome:
        """Mutate a chromosome using linear mutation

        Args:
            chrom (Chromosome):
            n (float): mutate linear, 0 for no linear mutation

        Returns: The mutated chromosome (Chromosome)

        """
        res = Chromosome(np.copy(chrom.data))

        mask = np.random.rand(self.__width_x, self.__width_y,
                              self.__color_range) > self.__mutation_prob

        border = self.__mutation_step
        if self.__mutation_linear:
            border = self.__mutation_step * 1 - n / self.__max_gen
        rnd_matrix = np.random.uniform(-1 * border, border, res.data.shape)
        added = np.add(res.data, rnd_matrix)
        res.data[mask] = added[mask]
        return res

    def combine(self, chrom1: Chromosome, chrom2: Chromosome) -> Chromosome:
        """Combine two good chromosomes randomly

        Args:
            chrom1 (Chromosome): First chromosome
            chrom2 (Chromosome): Second chromosome

        Returns: Combined chromosome (Chromosome)

        """
        # for each pixel choose value random from chrom1 or chrom2
        chrom = Chromosome(np.copy(chrom1.data))
        mask = np.random.randint(0, 2, size=chrom1.data.shape).astype(np.bool)
        chrom.data[mask] = chrom2.data[mask]
        return chrom
