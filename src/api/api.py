import os
import json
import requests
import time
from typing import List

import numpy as np
from io import BytesIO

from src.utils.image_utils import save_image


class ApiConnector:
    """Class to connect to the api, initially loading the configuration from file.

    """

    def __init__(self):
        """
        Load the api url and key from configuration file.
        """
        cfg_path = os.path.join(os.path.dirname(__file__), "../../config.json")
        with open(cfg_path, "r", encoding="utf8") as f:
            data = json.load(f)
        self._url = data["api_url"]
        self._api_key = data["api_key"]

    def send_query(self, image: np.ndarray) -> (List[str], np.ndarray):
        """
        Sends an image to the server.
        Args:
            image (numpy.array): The image to be sent. Shape must be either (64, 64, 1) if grayscale
            or (64, 64, 3) if RGB.

        Returns:
            The predicted class labels (np.array), the corresponding confidence values (np.array).

        Raises:
            ValueError: Invalid image
            ConnectionError: Bad api status code

        """
        if image.shape not in {(64, 64, 1), (64, 64, 3)}:
            raise ValueError(f'invalid image with shape {image.shape}')

        # create pseudo rgb image for api
        if image.shape[2] == 1:
            image = np.squeeze(image)
            image = np.stack((image,) * 3, axis=-1)

        # create file object from image
        image_file = BytesIO()

        save_image(image_file, image)
        image_file.seek(0)

        # send post requests
        response = requests.post(
            self._url, params={'key': self._api_key}, files={'image': image_file})

        while response.status_code == 429 or response.status_code == 400:
            print("no api capacities (response code:" + str(response.status_code) + "), waiting...")
            image_file.seek(0)
            time.sleep(5)
            response = requests.post(
                self._url, params={'key': self._api_key}, files={'image': image_file})

        # process response
        if response.status_code == 200:  # OK
            predictions = response.json()
            classes = []
            values = []
            for d in predictions:
                classes.append(d["class"])
                values.append(d["confidence"])

            return classes, values
        else:
            print(response.status_code)
            print(response.content)
            raise ConnectionError
