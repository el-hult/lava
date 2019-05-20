import numpy as np

class LavaBase:
    """ Base class for Lava models

    This is the class containing base functions for training a model.
    Do subclass this model freely!

    """

    def __init__(self):
        """ Create a new object of the lava class """
        self.y_history = None
        self.u_history = None

    def step(self, y, u):
        """ Perform one step of learning

        Args:
            y (ndarray): observed output
            u (ndarray): observed input
        """
        if self.y_history is None or self.u_history is None:
            self.u_history = u
            self.y_history = y
        else:
            self.u_history = np.column_stack([self.u_history, u])
            self.y_history = np.column_stack([self.y_history, y])


class InterceptRegressor:
    """ A class for returning a constant 1, useful for modelling intercepts."""
    def __init__(self):
        pass

    # noinspection PyMethodMayBeStatic
    def get_latent_regressor(self, y_history, u_history):
        return np.array([1])
