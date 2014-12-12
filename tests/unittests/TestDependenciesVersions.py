import unittest


class TestNumpyVersion(unittest.TestCase):

    def test_for_1_8_1(self):
        import numpy
        version = numpy.__version__
        self.assertEqual(version, "1.8.1")


class TestScikitLearnVersion(unittest.TestCase):

    def test_for_0_15_1(self):
        import sklearn
        version = sklearn.__version__
        self.assertEqual(version, '0.15.1')


class TestScipyVersion(unittest.TestCase):

    def test_for_0_14_0(self):
        import scipy
        version = scipy.__version__
        self.assertEqual(version, "0.14.0")