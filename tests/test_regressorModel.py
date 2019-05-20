from unittest import TestCase

from lava import RegressorModel


class TestRegressorModel(TestCase):

    def setUp(self) -> None:
        self.rm = RegressorModel()

    def test_get_regressor(self):
        self.assertRaises( NotImplementedError,self.rm.get_regressor,None,None)

    def test_get_regressor_stepwise(self):
        self.assertRaises( NotImplementedError,self.rm.get_regressor_stepwise,None,None)
