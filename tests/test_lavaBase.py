import unittest

from lava import LavaBase


class TestCausalEstimator(unittest.TestCase):

    def test_init(self):
        lb = LavaBase()
        self.assertEqual(lb.x,1)

    def test_step(self):
        lb = LavaBase()
        self.assertEqual(lb.step(3,6),3+6+1)
