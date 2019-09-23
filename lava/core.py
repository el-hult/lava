# Standard library import
import math

# Third party import
import numpy as np


# noinspection PyPep8Naming
class Lava:
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

    def step(self, y, u, n_recursive_rounds=3):
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

        if np.isscalar(u):
            u = np.asarray(u).reshape((1,))
        if np.isscalar(y):
            y = np.asarray(y).reshape((1,))

        n_y = len(y)
        n_u = len(u)
        if self.y_history is None or self.u_history is None:
            self.y_history = np.array([]).reshape(n_y, 0)
            self.u_history = np.array([]).reshape(n_u, 0)
        self.y_history = np.column_stack([y, self.y_history])
        self.u_history = np.column_stack([u, self.u_history])

        phi = self.nominal_model.current_regressor
        gamma = self.latent_model.current_regressor
        regressor_model_needs_more_data = phi is None or gamma is None
        if regressor_model_needs_more_data:
            self.nominal_model.update_regressor_stepwise(y, u)
            self.latent_model.update_regressor_stepwise(y, u)
            return False

        n_phi = phi.size
        n_gamma = gamma.size

        if self._steps_taken is None:
            self.initialize_identification_parameters(n_phi, n_gamma, n_y)

        Z_check = self.Z

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
                    alpha = eta + T[j, j] * Z_check[i, j] ** 2 + 2 * zeta[j] * Z_check[i, j]
                    g = zeta[j] + T[j, j] * Z_check[i, j]
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

                    z_diff = Z_check[i, j] - z_hat_ij
                    eta += T[j, j] * z_diff ** 2 + 2 * z_diff * zeta[j]
                    zeta += T[:, j] * z_diff
                    Z_check[i, j] = z_hat_ij

        self.Theta = self.Theta_bar - self.Z @ self.H.T
        self.Z = Z_check

        self.nominal_model.update_regressor_stepwise(y, u)
        nominal_regressor_vector = self.nominal_model.current_regressor
        self.latent_model.update_regressor_stepwise(y, u, nominal_regressor_vector)

        return self.Theta, self.Z

    def simulate(self, u):
        """Simulates the system for the duration of the input vector.

        Do note that since latent and nominal models may rely on historical data, this method assumes that the
        estimation starts where the training stopped.

        Args:
            u (ndarray): an array of inputs

        """
        n_time_steps = u.shape[1] if len(u.shape) == 2 else u.size
        n_output_dimension = self.Theta.shape[0]
        y_hat = np.zeros((n_output_dimension, n_time_steps))
        Theta_phi = np.zeros((n_output_dimension, n_time_steps))
        Z_gamma = np.zeros((n_output_dimension, n_time_steps))

        # first prediction is made on the last training sample!
        phi = self.nominal_model.current_regressor
        gamma = self.latent_model.current_regressor

        Theta_phi_f = self.Theta @ phi
        Z_gamma_f = self.Z @ gamma
        y_hat_f = Theta_phi_f + Z_gamma_f

        y_hat[:, 0] = y_hat_f
        Theta_phi[:, 0] = Theta_phi_f
        Z_gamma[:, 0] = Z_gamma_f

        for t in range(0, n_time_steps - 1):
            u_now = u[..., t]
            y_now = y_hat[..., t]
            phi = self.nominal_model.update_regressor_stepwise(y=y_now, u=u_now)
            gamma = self.latent_model.update_regressor_stepwise(y=y_now, u=u_now)

            Theta_phi_f = self.Theta @ phi
            Z_gamma_f = self.Z @ gamma
            y_hat_f = Theta_phi_f + Z_gamma_f

            # update forecast
            y_hat[:, t + 1] = y_hat_f
            Theta_phi[:, t + 1] = Theta_phi_f
            Z_gamma[:, t + 1] = Z_gamma_f

        return y_hat, Theta_phi, Z_gamma


class RegressorModel:
    def __init__(self):
        """Base class indicating what methods are needed for regressor models

        Do note that this class maybe could be put as a ABC class, but I'm not sure whether that is a good idea...

        """
        self.current_regressor = None
        """np.ndarray: The current state of the RegressorModel."""

    def update_regressor(self, y_history, u_history, nominal_regressor=None):
        """ Get a regressor vector based on historical observations

        Needs to be implemented by all RegressorModels"""
        raise NotImplementedError

    def update_regressor_stepwise(self, y: np.ndarray, u: np.ndarray, nominal_regressor=None):
        """" Get a regressor vector, only supplying the new observations.

        Needs to be implemented by all RegressorModels"""
        raise NotImplementedError


class InterceptRegressor(RegressorModel):
    def __init__(self):
        """ A class for returning a constant 1, useful for modelling intercepts."""
        super().__init__()
        self.current_regressor = np.ones(1)

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def update_regressor(self, y_history, u_history, nominal_regressor=None):
        return self.current_regressor

    def update_regressor_stepwise(self, y, u, nominal_regressor=None):
        return self.current_regressor


class ARXRegressor(RegressorModel):

    def __init__(self, y_lag_max, u_lag_max, y_lag_min=1, u_lag_min=1):
        """ Produce a AR model with lagged inputs, outputs, and an intercept

        The model is
            y(t+1) = SUM(A(k)*y(t-k)) + SUM(B(l)*u(t-l))
        where k ranges from y_lag_min to y_lag_max and l ranges similarly

        Example:
            arx = ARXRegressor(1,1)
            produces a AR(1) model in both signal and input like
            y(t) = A(1)*y(t-1) + B(1)*u(t-1)

            arx = ARXRegressor(3,2,1,2)
            produces a AR(1) model in both signal and input like
            y(t) = A(1)*y(t-1) + A(2)*y(t-2)
                  +A(3)*y(t-3) + B(2)*u(t-2)

        Args:
            y_lag_max: the most lagged order of output
            u_lag_max: the most lagged order of input
            y_lag_min: the least lagged order of output
            u_lag_min: the least lagged order of input

        """
        super().__init__()
        assert y_lag_min >= 1 and u_lag_min >= 1  # lag 0 cannot be used in predictions
        assert y_lag_max >= 0 and u_lag_max >= 0  # max lag 0 means NO autoregressor component
        self.y_lag_max = y_lag_max
        self.u_lag_max = u_lag_max
        self.y_lag_min = y_lag_min
        self.u_lag_min = u_lag_min

        # Below values will be set by first seen data
        self._first_run = True
        self._us = None
        self._ys = None

    def update_regressor(self, y_history, u_history, nominal_regressor=None):
        """Produce the whole regressor-vector based on a sufficient history of observations

        Updates and returns the self.current_regressor variable.
        This assumes that all old historical ys and us may be overwritten.

        Args:
            y_history (ndarray): The historical outputs. Columns of old observations. y = [y(t-1) y(t-2) ... ]
            u_history (ndarray): The historical inputs. Columns of old observations. u = [u(t-1) u(t-2) ...]
            nominal_regressor: The current value of the nominal regressor vector. Not used in this regressor model.

        Returns: ndarray: regressor = [y(t-k1) ; y(t-k1-1) ...y(y-k2); u(t-l1); u(t-l1-1) ... u(t-l2) ; 1]
        historical in/outputs, starting with most recent ys, then most recent us, finally a constant 1. All in a
        1D-array. The k1,k2,l1,l2 are the min and max lags for input and output respectively.
        """

        self._ys = y_history[:, 0:self.y_lag_max]
        self._us = u_history[:, 0:self.u_lag_max]

        if self._first_run:
            self._first_run = False

        has_seen_enough_observations = self._us.shape[1] == self.u_lag_max and self._ys.shape[1] == self.y_lag_max
        if has_seen_enough_observations:
            arx_vector = np.concatenate(
                [self._ys[:, self.y_lag_min - 1:].flatten('F'), self._us[:, self.u_lag_min - 1:].flatten('F'),
                 np.ones(1)],
                axis=0)
            self.current_regressor = arx_vector
        else:
            self.current_regressor = None

        return self.current_regressor

    def update_regressor_stepwise(self, y: np.ndarray, u: np.ndarray, nominal_regressor=None):
        """ARX vector by supplying only one observation at a time

        Updates and returns the self.current_regressor variable.

        Args:
            y (ndarray): Output from last timestep y(t-1)
            u (ndarray): Input in last timestep u(t-1)
            nominal_regressor (ndarray): The current value of the nominal regressor vector. phi(t)
             Not used in this regressor model, but is required by the interface

        Returns:
            None if too few observations are seen, otherwise a regression vector.

        """
        if self._first_run:
            self._first_run = False
            self._us = np.array([]).reshape((u.size, 0))
            self._ys = np.array([]).reshape((y.size, 0))

        if self.y_lag_max == 0:
            pass
        elif self._ys.shape[1] + 1 <= self.y_lag_max:
            self._ys = np.column_stack([y, self._ys])
        else:
            self._ys = np.column_stack([y, self._ys[:, 0:-1]])

        if self.u_lag_max == 0:
            pass
        elif self._us.shape[1] + 1 <= self.u_lag_max:
            self._us = np.column_stack([u, self._us])
        else:
            self._us = np.column_stack([u, self._us[:, 0:-1]])

        has_seen_enough_observations = self._us.shape[1] == self.u_lag_max and self._ys.shape[1] == self.y_lag_max
        if has_seen_enough_observations:
            arx_vector = np.concatenate(
                [self._ys[:, self.y_lag_min - 1:].flatten('F'), self._us[:, self.u_lag_min - 1:].flatten('F'),
                 np.ones(1)],
                axis=0)
            self.current_regressor = arx_vector
        else:
            self.current_regressor = None

        return self.current_regressor


class FourierRegressor(RegressorModel):

    def __init__(self, fourier_order, periodicity_y, periodicity_u, lags_y=1, lags_u=1):
        """A regressor object for fourier series expanding the historical y's and u's.

        Assumes a structure like
            F_y[l,k,m] = [cos(pi*(m+1)*y(t-l)[k]/T_y[k]) ;
                    sin((pi*(m+1)*y(t-l)[k]/T_y[k]]
        where
            l in range(y_lag_max)
            k in range(y_dim)
            m in range(1,fourier_order)

        and then flattens row-wise. (first over l, then k, then m)

        Similarly for F_u. Then F_y and F_u are concatenated.

        Args:
            fourier_order: the maximal order to fourier expand. The maximal j.
            periodicity_y: a vector of periods for the output signal. May be taken to be the range of values in each
                dimension.
            periodicity_u: a vector of periods for the input signal. May be taken to be the range of values in each
                dimension.
            lags_y = the number of lagged outputs to include in regressor vector
            lags_u = the number of lagged inputs to include in regressor vector
        """
        super().__init__()
        self.fourier_order = fourier_order
        self.periods_y = np.asarray(periodicity_y).flatten()
        self.periods_u = np.asarray(periodicity_u).flatten()
        self.y_lag_max = lags_y
        self.u_lag_max = lags_u
        self.y_dim = self.periods_y.size
        self.u_dim = self.periods_u.size

        # Will be populated by data at calls to update_regressor and ..._stepwise
        self._ys = np.array([]).reshape(self.y_dim, 0)
        self._us = np.array([]).reshape(self.u_dim, 0)

    def update_regressor_stepwise(self, y, u, nominal_regressor=None):

        assert len(y) == self.y_dim, f"Output shape {y.shape} does not match given period dimension {self.y_dim}"
        assert len(u) == self.u_dim, "Input shape does not math given periods"

        y_history = np.column_stack([y, self._ys])
        u_history = np.column_stack([u, self._us])
        if y_history.shape[1] >= self.y_lag_max and u_history.shape[1] >= self.u_lag_max:
            self.update_regressor(y_history, u_history)
        else:
            self._ys = y_history
            self._us = u_history

        return self.current_regressor

    def update_regressor(self, y_history, u_history, nominal_regressor=None):

        assert y_history.ndim == 2, "Historical records must be matrices with one column per time"
        assert u_history.ndim == 2, "Historical records must be matrices with one column per time"

        self._ys = y_history[:, 0:self.y_lag_max]
        self._us = u_history[:, 0:self.u_lag_max]

        Fy = np.array([]).reshape(0, self.y_lag_max * 2)
        Fu = np.array([]).reshape(0, self.u_lag_max * 2)

        for f_order in range(self.fourier_order):
            # add one row for each dimension of y - do all historical y's at the same time
            for y_dim in range(self.y_dim):
                tmp1 = np.array([np.cos(np.pi * (f_order + 1) * self._ys[y_dim, ...] / self.periods_y[y_dim]),
                                 np.sin(np.pi * (f_order + 1) * self._ys[y_dim, ...] / self.periods_y[y_dim])])

                Fy = np.vstack([Fy, tmp1.flatten('F')])

            for u_dim in range(self.u_dim):
                tmp2 = np.array([np.cos(np.pi * (f_order + 1) * self._us[u_dim, ...] / self.periods_u[u_dim]),
                                 np.sin(np.pi * (f_order + 1) * self._us[u_dim, ...] / self.periods_u[u_dim])])

                Fu = np.vstack([Fu, tmp2.flatten('F')])

        # matrices -> gamma-vector
        self.current_regressor = np.concatenate([Fy.flatten('C'), Fu.flatten('C')], axis=0)
        return self.current_regressor
