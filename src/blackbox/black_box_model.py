import os
from typing import List
from tensorflow.python.keras.models import load_model
import numpy as np


class LocalModel:
    """
    Simulates the api but without the query limitation, using a local model.
    A path to a keras model (.h5) is required for initialization.

    """

    def __init__(self, model_path: str):
        """
        Loads an existing local keras model (.h5) from file

        Args:
            model_path (str): path to the model
        """

        self.__model = load_model(os.path.abspath(model_path))
        # save input shape e.g. (64,64,1)
        self.__model_input_shape = self.__model.layers[0].get_config()["batch_input_shape"][1:]

    def send_query(self, image) -> (List[int], np.ndarray):
        """
        Sends a query image to the local model and tries to map this image for the expected input
        shape.

        Args:
            image (np.ndarray): The image to be sent

        Returns:
            List of respective class indexes (List[int]), the corresponding confidence values (
            np.ndarray)
        Raises:
            ValueError: Invalid image shape

        """
        # check query image shape and compare to expected models input shape!
        if len(image.shape) != 3:
            raise ValueError("Passed array is not of the right shape")

        if self.__model_input_shape[2] == 1 and image.shape[2] == 3:
            # convert to grayscale using a formula
            image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            image = np.expand_dims(image, -1)
        if self.__model_input_shape[2] == 3 and image.shape[2] == 1:
            image = np.dstack((image, image, image))

        image = np.expand_dims(image, axis=0)  # flatten image

        pred = self.__model.predict_on_batch(image)
        pred = pred[0]
        response_dict = {}
        for i, prob in enumerate(pred):
            response_dict[i] = prob

        response_label = np.array(list(response_dict.keys()), dtype=int)
        response_conf = np.array(list(response_dict.values()), dtype=float)
        return response_label, response_conf
