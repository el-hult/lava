# Standard library imports
from unittest import TestCase

# 3rd party imports
import numpy as np
from numpy.testing import assert_array_almost_equal

# local imports
import lava.core as lava


class TestARXRegressor(TestCase):
    def test_get_regressor(self):
        """ Test if we can identify the system matrices in AR(1) models"""

        np.random.seed(0)

        arx_regressor = lava.ARXRegressor(y_lag_max=1, u_lag_max=1)
        intercept_regressor = lava.InterceptRegressor()
        lb = lava.LavaBase(nominal_model=arx_regressor, latent_model=intercept_regressor)

        # setup system matrices
        A = np.array([[.9, .2], [-.2, .9]])
        B = np.array([[1, 0], [0, 0.1]])

        # generate input
        n_datapoints = 100
        input = 4 * np.random.normal(0, 1, size=(2, n_datapoints))

        # Initial values
        output = np.zeros((2, n_datapoints))

        for t in range(n_datapoints - 1):
            # AR(1)-model in both state and signal
            output[:, t + 1] = A @ output[:, t] + B @ input[:, t]

        # train lava model
        for t in range(n_datapoints):
            lb.step(y=output[:, t], u=input[:, t])

        theta_hat, zeta_hat = lb.Theta, lb.Z
        A_hat = theta_hat[:, 0:2]
        B_hat = theta_hat[:, 2:4]

        assert_array_almost_equal(A, A_hat)
        assert_array_almost_equal(B, B_hat)
