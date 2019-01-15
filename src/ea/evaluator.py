import numpy as np
from typing import Dict, List

from src.blackbox import black_box_model as nn
from src.api.api import ApiConnector


class Evaluator:
    """
        The evaluator sends a given image either to the api or to an ensemble of local models.
    """

    def __init__(self, index_to_label: Dict, models_path: List[str]):

        self.__index_to_label = index_to_label
        self.__connector = ApiConnector()

        if models_path is not None:
            black_box_models = []
            for model_path in models_path:
                black_box_models.append(nn.LocalModel(model_path))
            self.__blackbox_models = black_box_models

    def eval_own_nn(self, image: np.ndarray, class_index: int) -> float:
        """
        Use our own models to evaluate. If more than model given, return the mean confidence for
        a selected class.

        Args:
            image (np.ndarray): image to evaluate
            class_index (int): class index to evaluate

        Returns: confidence value (float)

        """
        img = image.astype(np.float32)

        conf_values = []
        for model in self.__blackbox_models:
            # get the class indexes and respective confidence values descending
            labels, confs = model.send_query(img)

            # or ignore classes and just take best conf from each nn
            if class_index == -1:
                max_conf = max(confs)
                conf_values.append(max_conf)
            elif class_index >= 0 and class_index < labels.size:
                conf_values.append(confs[class_index])
            else:
                raise IndexError("Class index is not encoded in the response")

        return sum(conf_values) / len(conf_values)

    def eval(self, image: np.ndarray, class_index: int) -> float:
        """
        Evaluates a given image (chromosome) for a specific class to the api.


        Args:
            image (np.ndarray): image to be evaluated
            class_index (int): class index to evaluate

        Returns: confidence value (float)

        """
        img = image.clip(0, 1).astype(np.float32)
        img = np.floor(img * 255) / 255

        labels, confs = self.__connector.send_query(img)
        if class_index == -1:
            max_conf = max(confs)
            return max_conf
        else:
            for i, label in enumerate(labels):
                if label == self.__index_to_label[str(class_index)]:
                    return confs[i]

        return -1
