import unittest
import numpy as np

from src.api import api

class TestApi(unittest.TestCase):

    # test the send query function
    # send a numpy array to the api
    # return top 5 results
    def test_send_query(self):
        image = np.random.rand(64, 64, 3)
        connection = api.ApiConnector()
        confs, labels = connection.send_query(image)

        # not a real test, but maybe nice to know if something changes
        self.assertEqual(len(confs), 5, msg="API response has changed!")
