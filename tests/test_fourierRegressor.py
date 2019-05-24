from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal


class TestFourierRegressor(TestCase):
    def test_update_regressor(self):
        """No error messages and some components are okay"""
        from lava import FourierRegressor

        ys = np.array([[1, 2, 3, 4, 1, 2, 3, 4],
                       [1, 2, 3, 4, 1, 2, 3, 4]])
        us = np.array([[1, 1, 1, 1, 0, 0, 0, 0]])
        T_ys = [4, 3]
        T_us = 2

        fr = FourierRegressor(fourier_order=3, periodicity_y=T_ys, periodicity_u=T_us, lags_y=3, lags_u=2)
        fr.update_regressor(ys, us)

        ans0 = np.cos(np.pi * 1 * ys[0, 0] / T_ys[0])
        self.assertEqual(ans0, fr.current_regressor[0])

        ans15 = np.sin(np.pi * 2 * ys[0, 1] / T_ys[0])
        self.assertEqual(ans15, fr.current_regressor[2 * 3 * 2 + 3])

    def test_update_regressor(self):
        """Only gives output when it should"""
        from lava import FourierRegressor

        y = np.array([1])
        u = np.array([0])
        T_ys = 1
        T_us = 1

        fr = FourierRegressor(fourier_order=1, periodicity_y=T_ys, periodicity_u=T_us, lags_y=3, lags_u=1)
        fr.update_regressor_stepwise(y, u)
        self.assertIsNone(fr.current_regressor)
        fr.update_regressor_stepwise(y, u)
        self.assertIsNone(fr.current_regressor)
        fr.update_regressor_stepwise(y, u)
        self.assertIsNotNone(fr.current_regressor)
