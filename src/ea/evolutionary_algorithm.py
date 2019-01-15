import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from skimage.measure import compare_ssim as ssim

from src.ea.chromosome import Chromosome
from src.ea.evaluator import Evaluator
from src.ea.poly import Poly
from src.ea.standard import Standard
from src.utils.image_utils import save_image, save_as_gif, get_filename


class EvolutionaryAlgorithm:
    """This class implements an Evolutionary Algorithm (EA) in order to evolve adversarial images.

    The EA evolves images by optimizing on the confidence value of a target model.


    """

    def __init__(self, class_index: List[int], models: List[str], index_label_dict: Dict,
                 ea_params: Dict,
                 color_range=3,
                 own_eval=True, poly=True, original=None):
        """
        Initializes the EA by some configuration parameters.

        Args:
            class_index (List[int]): list of class indexes for
            models (List[str]): list of paths to models
            index_label_dict (Dict): class index mapping to labels obtained by api
            ea_params (Dict): not changeable parameters by users
            color_range (int): either 1 for grayscale or 3 for rgb
            own_eval (bool): use local models for optimization
            poly (bool): use polygons for image generations
            original (np.ndarray): input image  initialization
        """

        self._pop_size = ea_params["pop_size"]
        self._copy = ea_params["copy"]
        self._random = ea_params["random"]
        self._mutate_only = ea_params["mutate_only"]
        self._max_gen = ea_params["max_gen"]
        self._min_conf = ea_params["min_conf"]
        self._width_x = ea_params["width_x"]
        self._width_y = ea_params["width_y"]
        self._mutation_prob = ea_params["mutation_prob"]
        self._mutation_step = ea_params["mutation_step"]
        self._mutation_linear = ea_params["mutation_linear"]

        if not own_eval:
            models = ""
            self._pop_size = 1
            self._copy = 0
            self._random = 1
            self._mutate_only = 0

        self._class_index = class_index
        self._own_eval = own_eval
        self._original = original
        self._evaluator = Evaluator(index_label_dict, models)

        # initialize the class that implements create chromo, mutate, combine, etc.
        if poly:
            self._ea = Poly(self._width_x, self._width_y, color_range)
        else:
            self._ea = Standard(self._width_x, self._width_y, color_range, self._mutation_prob,
                                self._mutation_step, self._mutation_linear, self._max_gen)

        if original is not None:
            self._original = original
            if not self._own_eval:
                self._mutate_only = self._random
            self._random = 0

        self._combine = self._pop_size - self._copy - self._random - self._mutate_only

    @property
    def class_index(self):
        return self._class_index

    @class_index.setter
    def class_index(self, class_index: int):
        self._class_index = class_index

    def fitness(self, data: np.ndarray, own=True) -> float:
        """
        Calculate the fitness of an image. How good is a solution, crucial for evolution of the EA

        Args:
            data (np.ndarray): Raw pixel values
            own (bool): Evaluate own local model(s)

        Returns:
            fitness value (float)

        """
        if self._own_eval and own:
            conf = self._evaluator.eval_own_nn(data, self._class_index)
        else:
            conf = self._evaluator.eval(data, self._class_index)

        image_diff = 1.0
        if self._original is not None:
            img = data.clip(0, 1).astype(np.float32)
            image_diff = ssim(self._original, img, multichannel=True, data_range=1.0)
            if image_diff > 0.05:
                image_diff = image_diff * 1000
            else:
                image_diff = 1.0

        if conf == -1:
            image_diff = 1.0
        return conf / image_diff

    def survival_of_the_fittest(self, generation: List[Chromosome]):
        """Select a good chromosome

        Take the best chromosome out of a randomly chosen set.

        Args:
            generation (List[Chromosome]): List of chromosomes

        Returns: The best randomly chosen chromosome (List[Chromosome])

        """
        champion = None
        best_fit = -2
        for i in range(5):
            candidate = generation[np.random.randint(0, self._pop_size - 1)]
            if candidate.fitness > best_fit:
                champion = candidate
                best_fit = candidate.fitness

        return champion

    def _combine_chrom(self, chrom1: Chromosome, chrom2: Chromosome) -> Chromosome:
        """Combine two chromosomes

        For each pixel choose value random from chromosome 1 or chromosome 2.

        Args:
            chrom1 (Chromosome): First chromosome
            chrom2 (Chromosome: Second chromosome

        Returns: Combined chromosome (Chromosome)

        """
        chrom = Chromosome(data=np.copy(chrom1.data))
        mask = np.random.randint(0, 2, size=chrom1.data.shape).astype(np.bool)
        chrom.data[mask] = chrom2.data[mask]

        return chrom

    def _create_first_generation(self, img=None) -> List[Chromosome]:
        """Initialize first generation

        Consists of pure random chromosomes.

        Args:
            img (np.ndarray): Create the first generation based on a input image.

        Returns: List of random chromosomes

        """
        generation = []
        if img is None:
            for i in range(self._pop_size):
                chrom = self._create_random_chrom()
                generation.append(chrom)
        else:
            for i in range(self._pop_size):
                chrom = Chromosome(data=np.copy(img))
                chrom = self._ea.mutate(chrom, 0)
                chrom.fitness = self.fitness(chrom.data)
                generation.append(chrom)

        return generation

    def _evaluate_api(self, champion: Chromosome, api_champ: Chromosome) -> (Chromosome, bool):
        """Evaluate an image with api

        Evaluate a new solution with the api. If the solution is to bad, the whole generation is
        reseted. This is necessary since the models are different and we want to optimise on their
        api and not our own.

        Args:
            champion (Chromosome): Champion to be evaluated
            api_champ (Chromosome): Api_champ previous best solution on api

        Returns: New champion (Chromosome) and whether the generation has to be resetted (bool)

        """
        reset = False
        if self._own_eval:
            conf = self.fitness(champion.data, False)
        else:
            conf = champion.fitness
        new_champ = None
        print("[EA] api confidence:", conf)

        if self._original is not None:
            if conf == -1:
                reset = True
                return new_champ, reset

        if conf > api_champ.api_fitness:
            # if we use a 1,1 EA and a valid conf was found,
            # switch from creating a random chromosome to mutation
            if not self._own_eval and self._random > 0 and conf > 0:
                self._mutate_only = self._random
                self._random = 0

            api_champ.api_fitness = conf
            api_champ.data = np.copy(champion.data)
            api_champ.fitness = champion.fitness
            new_champ = api_champ
        elif api_champ.api_fitness - conf > 0.03 and api_champ.api_fitness > 0.03:
            print("Reset to image with confidence {0:.3f}%".format(100 * api_champ.api_fitness))
            reset = True

        return new_champ, reset

    def _combine_and_mutate(self, i: int, generation: List[Chromosome]) -> Chromosome:
        """Combines two chromosomes and mutates the result

        Args:
            i (int): Denotes the mutation range for linear mutation
            generation (List[Chromosome]): The current generation

        Returns: A combined and mutated chromosome (Chromosome)

        """
        chrom1 = self.survival_of_the_fittest(generation)
        chrom2 = self.survival_of_the_fittest(generation)

        while chrom1 is chrom2:
            chrom2 = self.survival_of_the_fittest(generation)

        chrom = self._combine_chrom(chrom1, chrom2)
        chrom = self._ea.mutate(chrom, i)

        chrom.fitness = self.fitness(chrom.data)
        return chrom

    def _create_random_chrom(self) -> Chromosome:
        """Create a random chromosome

        Returns: Random created chromsome (Chromosome)

        """
        chrom = self._ea.create_random_chromosome()
        chrom.fitness = self.fitness(chrom.data)
        return chrom

    def _mutate(self, i: int, chrom: Chromosome) -> Chromosome:
        """ Mutate a chromosome

        Args:
            i (int): Denotes the mutation range for linear mutation
            chrom (Chromosome): the chromosome

        Returns: The mutated chromosome (Chromosome)

        """
        chrom = self._ea.mutate(chrom, i)
        chrom.fitness = self.fitness(chrom.data)
        return chrom

    def run(self):
        self.show_conf()

        # execute until the ea returns a good confidence
        for i in range(5):
            img = self.run_ea()
            if img is not None:
                break

    def run_ea(self) -> np.ndarray:
        """Main evolutionary algorithm

        Generic EA with elitisim, combine and mutate. For selection, the best n is used.

        Returns: The final image with highest confidence (np.ndarray)

        """
        api_champion = Chromosome(data=None, fitness=-2, api_fitness=-2)
        generation = self._create_first_generation(self._original)
        generation.sort(key=lambda x: x.fitness, reverse=True)
        champion = generation[0]
        images = [np.copy(champion.data)]
        print("[EA] Generation 0 - best conf:",
              "{0:.3f}%".format(100 * champion.fitness))

        api_champion, reset = self._evaluate_api(champion, api_champion)
        if api_champion is None:
            api_champion = Chromosome(data=None, fitness=-2, api_fitness=-2)

        self._conf_our = []
        self._conf_api = []

        for i in range(1, self._max_gen + 1):
            generation = self.get_next_generation(generation, i)
            generation.sort(key=lambda x: x.fitness, reverse=True)

            champion = generation[0]
            images.append(np.copy(champion.data))
            print("[EA] Generation {}".format(i),
                  "- best conf: {0:.3f}%".format(100 * champion.fitness))

            # validation against the api
            new_champ, reset = self._evaluate_api(champion, api_champion)
            if reset:
                generation = self._create_first_generation(api_champion.data)
                generation[-1] = api_champion
            elif new_champ is None:
                generation[-1] = api_champion
            else:
                api_champion = new_champ

            self._conf_api.append(api_champion.api_fitness)
            self._conf_our.append(champion.fitness)

            # restart after 50 iterations if confidence is low
            if i == 100 and api_champion.api_fitness < 0.02 and self._original is None:
                return None

            if api_champion.api_fitness >= self._min_conf:
                break

        image = api_champion.data
        save_image(os.path.join("results", "output_" + str(self._class_index) + ".png"), image)
        fname = os.path.join("results", "output_" + str(self._class_index) + ".gif")
        save_as_gif(fname, images)

        return image

    def get_next_generation(self, generation: List[Chromosome], i: int) -> List[Chromosome]:
        """
        Returns the next generation of chromosomes using e.g. random created chromosomes,
        mutated chromosomes or combined and mutated chromosomes.

        Args:
            generation (List[Chromosome]): Current generation
            i (int): Current generation

        Returns: List of the new created population (List[Chromosome])

        """
        next_generation = []
        for j in range(self._combine):
            chrom = self._combine_and_mutate(i, generation)
            next_generation.append(chrom)

        for j in range(self._random):
            chrom = self._create_random_chrom()
            next_generation.append(chrom)

        for j in range(self._mutate_only):
            chrom = self._mutate(i, generation[j])
            next_generation.append(chrom)

        for j in range(self._copy):
            next_generation.append(generation[j])
        return next_generation

    def print_statistic(self):
        """Plots the confidence history over the generations

        """
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(0, len(self._conf_api) * 1, step=1), self._conf_api,
                 label="Target network")
        plt.plot(np.arange(0, len(self._conf_our)), self._conf_our, label="Local network(s)")
        plt.ylabel("Confidence")
        plt.xlabel("Generation")
        plt.legend()
        fname = "conf_over_generations_" + str(self._class_index) + ".svg"
        fname = get_filename(os.path.join("results", fname))
        plt.savefig(fname, type='svg')

    def show_conf(self):
        """Print current configuration

        Returns: Prints current configuration to console (None)

        """
        print("Pop size:", self._pop_size)
        print("Mutate only:", self._mutate_only)
        print("Mutation Probability:", self._mutation_prob)
        print("Mutation step:", self._mutation_step)
        print("Linear Mutation:", self._mutation_linear)
        print("Poly:", type(self._ea))
