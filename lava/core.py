import numpy as np


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



    def step(self, y, u):
        """ Perform one step of learning

        Args:
            y (ndarray): observed output at one point in time
            u (ndarray): observed input at one point in time
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

        # The model has provided some estimate
        return True


class RegressorModel:
    def __init__(self):
        """Base class indicating what methods are needed for regressor models

        Do note that this class maybe should be put as a ABC class, but I'm not sure whether that is a good idea...

        """
        pass

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

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def get_regressor(self, y_history, u_history, nominal_regressor=None):
        return np.ones(1)

    def get_regressor_stepwise(self, y, u, nominal_regressor=None):
        return np.ones(1)


class ARXRegressor(RegressorModel):

    def __init__(self, y_lag_order, u_lag_order):
        """ Produce a AR model with lagged inputs, outputs, and an intercept"""
        super().__init__()
        self.y_lags = y_lag_order
        self.u_lags = u_lag_order

        # Below values will be set by first seen data
        self._first_run = True
        self._n_y = None
        self._n_u = None
        self._us = None
        self._ys = None

    def get_regressor(self, y_history, u_history, nominal_regressor=None) -> np.ndarray:
        """Produce the whole regressor-vector based on a sufficient history of observations

        This assumes that all old historical ys and us may be overwritten.

        Args:
            y_history (ndarray): The historical outputs. Columns of old observations. y = [y(t-1) y(t-2) ... ]
            u_history (ndarray): The historical inputs. Columns of old observations. u = [u(t-1) u(t-2) ...]

        Returns: ndarray: regressor = [y(t-1) ; y(t-2) ... u(t-1); u(t-2) ... 1] historical in/outputs, starting with
        most recent ys, then most recent us, finally a constant 1. All in a 1D-array
        """

        if y_history.shape[1] < self.y_lags:
            raise ValueError(f"Not enought history presented. Y needs {self.y_lags} records - "
                             f"only {y_history.shape[1]} were presented.")

        if u_history.shape[1] < self.u_lags:
            raise ValueError(f"Not enought history presented. U needs {self.u_lags} records - "
                             f"only {u_history.shape[1]} were presented.")

        self._ys = y_history[:, 0:self.y_lags]
        self._us = u_history[:, 0:self.u_lags]

        if self._first_run:
            self._first_run = False
            self._n_u = u_history.shape[0]
            self._n_y = y_history.shape[0]

        arx_vector = np.concatenate(
            [self._ys.flatten('F'), self._us.flatten('F'), np.ones(1)]
            , axis=0)

        return arx_vector

    def get_regressor_stepwise(self, y: np.ndarray, u: np.ndarray, nominal_regressor: np.ndarray = None) -> np.ndarray:
        """Get a arx vector by supplying only one observation at a time

        if there is too little data to supply a arx regressor vector, None is returned

        Returns:
            None if too few observations are seen, otherwies a regression vector.

        """
        if self._first_run:
            self._first_run = False
            self._n_u = u.size
            self._n_y = y.size
            self._us = u[:, np.newaxis]
            self._ys = y[:, np.newaxis]
        else:

            if self._ys.shape[1] <= (self.y_lags - 1):
                self._ys = np.column_stack([y, self._ys])
            else:
                self._ys = np.column_stack([y, self._ys[:, 0:-1]])

            if self._us.shape[1] <= (self.u_lags - 1):
                self._us = np.column_stack([u, self._us])
            else:
                self._us = np.column_stack([u, self._us[:, 0:-1]])

        has_seen_enough_observations = self._us.shape[1] == self.u_lags and self._ys.shape[1] == self.y_lags
        if has_seen_enough_observations:
            arx_vector = np.concatenate(
                [self._ys.flatten('F'), self._us.flatten('F'), np.ones(1)]
                , axis=0)
            return arx_vector
        else:
            return None
