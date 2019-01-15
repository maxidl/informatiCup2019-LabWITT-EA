import numpy as np


class Chromosome:
    """Creates a chromosome containing data and the respective fitness value

    """

    def __init__(self, data=np.ndarray, fitness=-1, api_fitness=-2):
        """Set default fitness values and initialize data

        Args:
            data (np.ndarray): the image
            fitness (float): confidence value on local model(s)
            api_fitness (float): confidence value on target model
        """
        self.fitness = fitness
        self.api_fitness = api_fitness
        self.data = data

    def __str__(self):
        return "data_shape: {} conf: {}".format(self.data.shape, self.fitness)
