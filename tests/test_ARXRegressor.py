from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from lava import ARXRegressor


class TestARXRegressor(TestCase):
    def test_get_regressor(self):
        """ Test if min and max lags works for the not-stepwise regressor builder."""
        arx_regressor = ARXRegressor(y_lag_max=2, u_lag_max=1, y_lag_min=2,u_lag_min=1)

        y_hist = np.array([[1, 3, 2],
                           [4, 3, 2]])
        u_hist = np.array([[1, 3, 1],
                           [4, 3, 1]])

        phi = arx_regressor.update_regressor(y_hist, u_hist)
        assert_array_equal(np.array([2, 2, 3, 3, 1]), phi)

    def test_get_regressor_short_history(self):
        y_hist = np.array([[1, 3],
                           [4, 3]])
        u_hist = np.array([[1, 3],
                           [4, 3]])

        arx_regressor = ARXRegressor(y_lag_max=4, u_lag_max=1)
        self.assertRaises(ValueError, arx_regressor.update_regressor, y_hist, u_hist)

        arx_regressor = ARXRegressor(y_lag_max=1, u_lag_max=4)
        self.assertRaises(ValueError, arx_regressor.update_regressor, y_hist, u_hist)

    def test_get_regressor_stepwise(self):
        """ check if stepwise builder works with and witout defaults"""
        arx_regressor = ARXRegressor(y_lag_max=1, u_lag_max=0,y_lag_min=0,u_lag_min=0)
        phi = arx_regressor.update_regressor_stepwise(np.array([3, 3]), np.array([3, 3]))
        self.assertIsNone(phi)
        phi = arx_regressor.update_regressor_stepwise(np.array([1, 4]), np.array([1, 4]))
        assert_array_equal(np.array([1, 4, 3, 3, 1, 4, 1]), phi)
        phi = arx_regressor.update_regressor_stepwise(np.array([0, 0]), np.array([0, 0]))
        assert_array_equal(np.array([0, 0, 1, 4, 0, 0, 1]), phi)

        arx_regressor = ARXRegressor(y_lag_max=2, u_lag_max=1)
        phi = arx_regressor.update_regressor_stepwise(np.array([3, 3]), np.array([3, 3]))
        self.assertIsNone(phi)
        phi = arx_regressor.update_regressor_stepwise(np.array([1, 4]), np.array([1, 4]))
        self.assertIsNone(phi)
        phi = arx_regressor.update_regressor_stepwise(np.array([0, 0]), np.array([0, 0]))
        assert_array_equal(np.array([1, 4, 3, 3, 1, 4, 1]), phi)
        phi = arx_regressor.update_regressor_stepwise(np.array([9, 9]), np.array([9, 9]))
        assert_array_equal(np.array([0, 0, 1, 4, 0, 0, 1]), phi)
