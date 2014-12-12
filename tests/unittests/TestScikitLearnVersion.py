import unittest


class TestScikitLearnVersion(unittest.TestCase):

    def test_for_0_15_1(self):
        import sklearn
        version = sklearn.__version__
        self.assertEqual(version, '0.15.1')

