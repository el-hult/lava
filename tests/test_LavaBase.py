import unittest
import numpy as np

from lava import Lava, ARXRegressor, InterceptRegressor


class TestLavaBase(unittest.TestCase):

    def test_init(self):
        lb = Lava(InterceptRegressor(), InterceptRegressor())
        self.assertEqual(1, 1)

    def test_step_stacking(self):
        lb = Lava(InterceptRegressor(), InterceptRegressor())
        y = np.array([1, 4, 6]).T
        u = np.array([5, 4]).T
        lb.step(y, u)
        lb.step(y, u)

        self.assertEqual((2, 2), lb.u_history.shape, f"the u history has shape {lb.u_history.shape}")
        self.assertEqual((3, 2), lb.y_history.shape, f"the y history has shape {lb.y_history.shape}")

    def test_step_regularized_ARX(self):
        arx_regressor = ARXRegressor(y_lag_max=2, u_lag_max=1)
        intercept_regressor = InterceptRegressor()
        lb = Lava(nominal_model=intercept_regressor, latent_model=arx_regressor)

        ret = lb.step(np.array([3, 3]), np.array([3, 3]))
        self.assertFalse(ret)

        lb.step(np.array([1, 1]), np.array([2, 2]))
        ret = lb.step(np.array([1, 1]), np.array([2, 2]))
        self.assertTrue(ret)

    def test_simulate(self):
        """Make sure that the simulation loop works with the simplest of regressor models"""

        from lava import RegressorModel
        from numpy.testing import assert_array_almost_equal

        # noinspection PyAbstractClass
        i1 = InterceptRegressor()
        i2 = InterceptRegressor()

        lb = Lava(i1, i2)

        # inject trained parameters
        lb.Theta = np.array([[2], [0]])
        lb.Z = np.array([[1], [1]])

        y_true = np.array([[3, 3, 3], [1, 1, 1]])
        y_hat, *_ = lb.simulate(np.array([1,2,3]))
        assert_array_almost_equal(y_true, y_hat)
