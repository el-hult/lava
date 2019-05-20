import unittest
import numpy as np

from lava import LavaBase


class TestLavaBase(unittest.TestCase):

    def test_init(self):
        lb = LavaBase()
        self.assertEqual(1,1)

    def test_step_stacking(self):
        lb = LavaBase()
        y = np.array([1,4,6]).T
        u = np.array([5,4]).T
        lb.step(y, u)
        lb.step(y, u)

        self.assertEqual((2,2), lb.u_history.shape, f"the u history has shape {lb.u_history.shape}")
        self.assertEqual((3,2), lb.y_history.shape, f"the y history has shape {lb.y_history.shape}")
