import unittest
import numpy as np

from lava import LavaBase, ARXRegressor, InterceptRegressor


class TestLavaBase(unittest.TestCase):

    def test_init(self):
        lb = LavaBase(InterceptRegressor(),InterceptRegressor())
        self.assertEqual(1, 1)

    def test_step_stacking(self):
        lb = LavaBase(InterceptRegressor(),InterceptRegressor())
        y = np.array([1, 4, 6]).T
        u = np.array([5, 4]).T
        lb.step(y, u)
        lb.step(y, u)

        self.assertEqual((2, 2), lb.u_history.shape, f"the u history has shape {lb.u_history.shape}")
        self.assertEqual((3, 2), lb.y_history.shape, f"the y history has shape {lb.y_history.shape}")

    def test_step_regularized_ARX(self):
        arx_regressor = ARXRegressor(y_lag_max=2, u_lag_max=1)
        intercept_regressor = InterceptRegressor()
        lb = LavaBase(nominal_model=intercept_regressor, latent_model=arx_regressor)

        ret = lb.step(np.array([3, 3]), np.array([3, 3]))
        self.assertFalse(ret)

        ret = lb.step(np.array([1, 1]), np.array([2, 2]))
        self.assertFalse(ret)

        ret = lb.step(np.array([1, 4]), np.array([1, 4]))
        self.assertTrue(ret)

