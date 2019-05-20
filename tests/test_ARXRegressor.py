from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from lava import ARXRegressor


class TestARXRegressor(TestCase):
    def test_get_regressor(self):
        arx_regressor = ARXRegressor(y_lag_order=2, u_lag_order=1)

        y_hist = np.array([[1, 3],
                           [4, 3]])
        u_hist = np.array([[1, 3],
                           [4, 3]])

        phi = arx_regressor.get_regressor(y_hist, u_hist)
        assert_array_equal(np.array([1, 4, 3, 3, 1, 4, 1]), phi)
        self.assertEqual(phi.shape, (7,))

    def test_get_regressor_short_history(self):
        y_hist = np.array([[1, 3],
                           [4, 3]])
        u_hist = np.array([[1, 3],
                           [4, 3]])

        arx_regressor = ARXRegressor(y_lag_order=4, u_lag_order=1)
        self.assertRaises(ValueError, arx_regressor.get_regressor, y_hist, u_hist)

        arx_regressor = ARXRegressor(y_lag_order=1, u_lag_order=4)
        self.assertRaises(ValueError, arx_regressor.get_regressor, y_hist, u_hist)

    def test_get_regressor_stepwise(self):
        arx_regressor = ARXRegressor(y_lag_order=2, u_lag_order=1)

        phi = arx_regressor.get_regressor_stepwise(np.array([3, 3]), np.array([3, 3]))
        self.assertIsNone(phi)
        phi = arx_regressor.get_regressor_stepwise(np.array([1, 4]), np.array([1, 4]))
        assert_array_equal(np.array([1, 4, 3, 3, 1, 4, 1]), phi)
        phi = arx_regressor.get_regressor_stepwise(np.array([0, 0]), np.array([0, 0]))
        assert_array_equal(np.array([0, 0, 1, 4, 0, 0, 1]), phi)
