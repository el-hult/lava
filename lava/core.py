
class LavaBase:
    """ Base class for Lava models

    This is the class containing base functions for training a model.
    Do subclass this model freely!

    """

    def __init__(self):
        """
        Create a new object of the lava class

        Args:
            foo: some arg
        """
        self.x = 1


    def step(self, y, u):
        """ Dummy implementation of code to come"""
        return self.x+y+u
