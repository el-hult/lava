# Standard Libraries
from unittest import TestCase

# Third party imports
import numpy as np

# Local Imports
from lava import InterceptRegressor


class TestLatentIntercept(TestCase):
    def test_get_latent_regressor(self):
        li = InterceptRegressor()
        y_history = np.array([[1,1],[3,4]])
        u_history = np.array([[1, 1], [3, 4]])

        one = li.get_latent_regressor(y_history,u_history)
        self.assertEqual(np.array([1]),one)
