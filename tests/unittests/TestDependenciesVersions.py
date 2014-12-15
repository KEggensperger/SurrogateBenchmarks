import unittest


class TestDependenciesVersion(unittest.TestCase):

    def test_sklearn(self):
        import sklearn
        version = sklearn.__version__
        self.assertEqual(version, '0.15.1')

    def test_scipy(self):
        import scipy
        version = scipy.__version__
        self.assertEqual(version, "0.14.0")

    def test_numpy(self):
        import numpy
        version = numpy.__version__
        self.assertEqual(version, "1.8.1")