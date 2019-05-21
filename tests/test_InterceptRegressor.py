# Standard Libraries
from unittest import TestCase

# Third party imports
import numpy as np

# Local Imports
from lava import InterceptRegressor

class TestInterceptRegressor(TestCase):

    def test_get_regressor(self):
        li = InterceptRegressor()
        y_history = np.array([[1, 1], [3, 4]])
        u_history = np.array([[1, 1], [3, 4]])

        one = li.update_regressor(y_history, u_history)
        self.assertEqual(np.array([1]), one)

        one = li.update_regressor(y_history, u_history, nominal_regressor=None)
        self.assertEqual(np.array([1]), one)

    def test_get_regressor_stepwise(self):
        li = InterceptRegressor()
        one = li.update_regressor(None, None, None)
        self.assertEqual(np.array([1]), one)

