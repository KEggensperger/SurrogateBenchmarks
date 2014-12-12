import unittest


class TestScipyVersion(unittest.TestCase):

    def test_for_0_14_0(self):
        import scipy
        version = scipy.__version__
        self.assertEqual(version, "0.14.0")

