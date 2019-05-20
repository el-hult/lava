import numpy as np

class LavaBase:
    def __init__(self, nominal_model=None, latent_model=None):
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
            self.u_history = np.column_stack([self.u_history, u])
            self.y_history = np.column_stack([self.y_history, y])


class RegressorModel:
    def __init__(self):
        """Base class indicating what methods are needed for regressor models"""
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
        return np.array([1])

    def get_regressor_stepwise(self, y, u, nominal_regressor=None):
        return np.array([1])