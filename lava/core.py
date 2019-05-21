# Standard library import
import math

# Third party import
import numpy as np


# noinspection PyPep8Naming
class LavaBase:
    def __init__(self, nominal_model, latent_model):
        """ A Latent Variable estimator object

        This model contains methods for training and predicting using the LAVA method.

        Args:
            nominal_model (RegressorModel): the model that produces nominal variable vector phi
            latent_model (RegressorModel): the model that produces latent variable vector gamma
        """
        self.y_history = None
        self.u_history = None
        self.latent_model = latent_model
        self.nominal_model = nominal_model

        # state used in systems identification
        self._steps_taken = None
        self.Psi_phi_phi = None
        self.Psi_gamma_gamma = None
        self.Psi_y_y = None
        self.Psi_phi_gamma = None
        self.Psi_phi_y = None
        self.Psi_gamma_y = None
        self.Z = None
        self.Theta_bar = None
        self.P = None
        self.H = None

        # variable for estimation and prediction
        self.Theta = None

    def initialize_identification_parameters(self, n_phi, n_gamma, n_y):
        """Resets identification parameters to default values.

        Args:
            n_gamma (int): the dimension of the latent variable space
            n_phi (int): the dimension of the nominal model space
            n_y (int): the dimension of the output space

        """
        self._steps_taken = 0
        self.Psi_phi_phi = np.zeros((n_phi, n_phi))
        self.Psi_gamma_gamma = np.zeros((n_gamma, n_gamma))
        self.Psi_y_y = np.zeros((n_y, n_y))
        self.Psi_phi_gamma = np.zeros((n_phi, n_gamma))
        self.Psi_phi_y = np.zeros((n_phi, n_y))
        self.Psi_gamma_y = np.zeros((n_gamma, n_y))
        self.Z = np.zeros((n_y, n_gamma))
        self.Theta_bar = np.zeros((n_y, n_phi))
        self.P = 100000 * np.eye(n_phi)
        self.H = np.zeros((n_phi, n_gamma))

    def step(self, y, u, n_recursive_rounds=3) -> (np.ndarray, np.ndarray):
        """ Perform one step of learning

        Args:
            y (ndarray): observed output at one point in time
            u (ndarray): observed input at one point in time
            n_recursive_rounds (int): number of iterations in recursive identification

        Returns:
            Current estimates of Theta and Zeta, if there is enough data. Otherwise returns None.

        * Update vector of products of inputs and outputs :math:`\Psi`
        * Update covariance matrix :math:`P`
        * Update regression matrix :math:`H`
        * Update matrix of parameters for nominal model :math:`\overline{\Theta}`
        * Update matrix of parameters for latent variable model :math:`Z`
        """
        if self.y_history is None or self.u_history is None:
            self.u_history = u
            self.y_history = y
        else:
            self.u_history = np.column_stack([u, self.u_history])
            self.y_history = np.column_stack([y, self.y_history])

        phi = self.nominal_model.get_regressor_stepwise(y, u)
        gamma = self.latent_model.get_regressor_stepwise(y, u)

        regressor_model_needs_more_data = phi is None or gamma is None
        if regressor_model_needs_more_data:
            return False

        n_y = y.size
        n_phi = phi.size
        n_gamma = gamma.size

        if self._steps_taken is None:
            self.initialize_identification_parameters(n_phi, n_gamma, n_y)

        Z_brows = self.Z

        # 2. Update P, \bar \Theta, H
        self.P -= (self.P @ np.outer(phi, phi) @ self.P) / (
                1 + phi @ self.P @ phi
        )
        self.Theta_bar += np.outer(y - self.Theta_bar @ phi, phi) @ self.P
        self.H += self.P @ (
                np.outer(phi, gamma) - np.outer(phi, phi) @ self.H
        )

        # 3. Update psi matrices according to eq (58)
        self.Psi_phi_phi += np.outer(phi, phi)
        self.Psi_gamma_gamma += np.outer(gamma, gamma)
        self.Psi_y_y += np.outer(y, y)
        self.Psi_phi_gamma += np.outer(phi, gamma)
        self.Psi_phi_y += np.outer(phi, y)
        self.Psi_gamma_y += np.outer(gamma, y)

        # 4. Update the T-matrix.
        T = (
                self.Psi_gamma_gamma
                - self.Psi_phi_gamma.T @ self.H
                - self.H.T @ self.Psi_phi_gamma
                + self.H.T @ self.Psi_phi_phi @ self.H
        )

        # increment step
        self._steps_taken += 1

        for i in range(n_y):
            kappa = (
                    self.Psi_y_y[i, i]
                    + self.Theta_bar[i, :] @ self.Psi_phi_phi @ self.Theta_bar[i, :].T
                    - 2 * self.Theta_bar[i, :] @ self.Psi_phi_y[:, i]
            )

            rho = (
                    self.Psi_gamma_y[:, i]
                    - self.Psi_phi_gamma.T @ self.Theta_bar[i, :].T
                    - self.H.T @ self.Psi_phi_y[:, i]
                    + self.H.T @ self.Psi_phi_phi @ self.Theta_bar[i, :].T
            )

            eta = kappa - 2 * rho.T @ self.Z[i, :].T + self.Z[i, :] @ T @ self.Z[i, :].T
            zeta = rho - T @ self.Z[i, :].T

            for k in range(n_recursive_rounds):
                for j in range(n_gamma):
                    alpha = eta + T[j, j] * Z_brows[i, j] ** 2 + 2 * zeta[j] * Z_brows[i, j]
                    g = zeta[j] + T[j, j] * Z_brows[i, j]
                    beta = T[j, j]
                    w = np.sqrt(self.Psi_gamma_gamma[j, j] / self._steps_taken)

                    if alpha * w ** 2 < g ** 2:
                        try:
                            r_hat = np.abs(g) / beta - w / (beta * math.sqrt(beta - w ** 2)) \
                                    * math.sqrt(alpha * beta - g ** 2)
                            z_hat_ij = np.sign(g) * r_hat
                        except ValueError:
                            # This can only happen due to numerical round-off errors.
                            # It should be z_hat_ij = 0 in this case
                            z_hat_ij = 0
                    else:
                        z_hat_ij = 0

                    z_diff = Z_brows[i, j] - z_hat_ij
                    eta += T[j, j] * z_diff ** 2 + 2 * z_diff * zeta[j]
                    zeta += T[:, j] * z_diff
                    Z_brows[i, j] = z_hat_ij

        self.Theta = self.Theta_bar - self.Z @ self.H.T
        self.Z = Z_brows

        return self.Theta, self.Z


class RegressorModel:
    def __init__(self):
        """Base class indicating what methods are needed for regressor models

        Do note that this class maybe should be put as a ABC class, but I'm not sure whether that is a good idea...

        """

    def get_regressor(self, y_history, u_history, nominal_regressor=None):
        """ Get a regressor vector based on historical observations

        Needs to be implemented by all RegressorModels"""
        raise NotImplementedError

    def get_regressor_stepwise(self, y, u, nominal_regressor=None):
        """" Get a regressor vector, only suplying the new observations.

        Needs to be implemented by all RegressorModels"""
        raise NotImplementedError


class InterceptRegressor(RegressorModel):
    def __init__(self):
        """ A class for returning a constant 1, useful for modelling intercepts."""
        super().__init__()
        self.one = np.ones(1)

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def get_regressor(self, y_history, u_history, nominal_regressor=None):
        return self.one

    def get_regressor_stepwise(self, y, u, nominal_regressor=None):
        return self.one


class ARXRegressor(RegressorModel):

    def __init__(self, y_lag_max, u_lag_max, y_lag_min=1, u_lag_min=1):
        """ Produce a AR model with lagged inputs, outputs, and an intercept

        The model is
            y(t) = SUM(A(k)*y(t-k)) + SUM(B(l)*u(t-l))
        where k ranges from y_lag_min to y_lag_max nad l ranges similarly

        Example:
            arx = ARXRegressor(1,1)
            produces a AR(1) model in both signal and input

        Args:
            y_lag_max: the most lagged order of output
            u_lag_max: the most lagged order of input
            y_lag_min: the least lagged order of output
            u_lag_min: the least lagged order of input

        """
        super().__init__()
        self.y_lag_max = y_lag_max
        self.u_lag_max = u_lag_max
        self.y_lag_min = y_lag_min
        self.u_lag_min = u_lag_min

        # Below values will be set by first seen data
        self._first_run = True
        self._n_y = None
        self._n_u = None
        self._us = None
        self._ys = None

        # Will be set when enough data has passed
        self._current_state = None

    def get_regressor(self, y_history, u_history, nominal_regressor=None) -> np.ndarray:
        """Produce the whole regressor-vector based on a sufficient history of observations

        This assumes that all old historical ys and us may be overwritten.

        Args:
            y_history (ndarray): The historical outputs. Columns of old observations. y = [y(t-1) y(t-2) ... ]
            u_history (ndarray): The historical inputs. Columns of old observations. u = [u(t-1) u(t-2) ...]
            nominal_regressor: The current value of the nominal regressor vector. Not used in this regressor model.

        Returns: ndarray: regressor = [y(t-k1) ; y(t-k1-1) ...y(y-k2); u(t-l1); u(t-l1-1) ... u(t-l2) ; 1]
        historical in/outputs, starting with most recent ys, then most recent us, finally a constant 1. All in a
        1D-array. The k1,k2,l1,l2 are the min and max lags for input and output respectively.
        """

        if y_history.shape[1] - 1 < self.y_lag_max:
            raise ValueError(f"Not enough history presented. Y needs {self.y_lag_max} records - "
                             f"only {y_history.shape[1]} were presented.")

        if u_history.shape[1] - 1 < self.u_lag_max:
            raise ValueError(f"Not enough history presented. U needs {self.u_lag_max} records - "
                             f"only {u_history.shape[1]} were presented.")

        self._ys = y_history[:, 0:self.y_lag_max + 1]
        self._us = u_history[:, 0:self.u_lag_max + 1]

        if self._first_run:
            self._first_run = False
            self._n_u = u_history.shape[0]
            self._n_y = y_history.shape[0]

        arx_vector = np.concatenate(
            [self._ys[:, self.y_lag_min:].flatten('F'), self._us[:, self.u_lag_min:].flatten('F'), np.ones(1)],
            axis=0)

        return arx_vector

    def get_regressor_stepwise(self, y, u, nominal_regressor=None):
        """Get a arx vector by supplying only one observation at a time

        if there is too little data to supply an arx regressor vector, None is returned

        Args:
            y (ndarray): The historical outputs. Columns of old observations. y = [y(t-1) y(t-2) ... ]
            u (ndarray): The historical inputs. Columns of old observations. u = [u(t-1) u(t-2) ...]
            nominal_regressor (ndarray): The current value of the nominal regressor vector. Not used in this regressor
             model, but is required by the interface

        Returns:
            None if too few observations are seen, otherwise a regression vector.

        """
        if self._first_run:
            self._first_run = False
            self._n_u = u.size
            self._n_y = y.size
            self._us = u[:, np.newaxis]
            self._ys = y[:, np.newaxis]
        else:

            if self._ys.shape[1] <= self.y_lag_max:
                self._ys = np.column_stack([y, self._ys])
            else:
                self._ys = np.column_stack([y, self._ys[:, 0:-1]])

            if self._us.shape[1] <= self.u_lag_max:
                self._us = np.column_stack([u, self._us])
            else:
                self._us = np.column_stack([u, self._us[:, 0:-1]])

        has_seen_enough_observations = self._us.shape[1] - 1 == self.u_lag_max and self._ys.shape[
            1] - 1 == self.y_lag_max
        if has_seen_enough_observations:
            arx_vector = np.concatenate(
                [self._ys[:, self.y_lag_min:].flatten('F'), self._us[:, self.y_lag_min:].flatten('F'), np.ones(1)],
                axis=0)
            self._current_state = arx_vector
        else:
            self._current_state = None

        return self._current_state
