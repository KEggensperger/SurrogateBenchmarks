import unittest

#import unittests.ArcGPTest
import unittests.GaussianProcessTest
import unittests.LinearRegressionTest
import unittests.RandomForestTest
import unittests.ModelUtilTest
import unittests.RidgeRegressionTest
import unittests.ScalerTest
import unittests.SupportVectorTest
import unittests.RemoveInactive
import unittests.TestDependenciesVersions
import unittests.KNNTest


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(unittest.makeSuite(unittests.TestDependenciesVersions.
                                      TestDependenciesVersion))
    _suite.addTest(unittest.makeSuite(unittests.KNNTest.KNNTest))
    """
    for i in range(2):
        #_suite.addTest(unittest.makeSuite(unittests.ArcGPTest.ArcGPTest))
        _suite.addTest(unittest.makeSuite(unittests.RandomForestTest.
                                          RandomForestTest))
        _suite.addTest(unittest.makeSuite(unittests.RidgeRegressionTest.
                                          RidgeRegressionTest))
        _suite.addTest(unittest.makeSuite(unittests.GaussianProcessTest.
                                          GaussianProcessTest))
        _suite.addTest(unittest.makeSuite(unittests.LinearRegressionTest.
                                          LinearRegressionTest))
        _suite.addTest(unittest.makeSuite(unittests.ModelUtilTest.
                                          ModelUtilTest))
        _suite.addTest(unittest.makeSuite(unittests.ScalerTest.ScalerTest))
        _suite.addTest(unittest.makeSuite(unittests.SupportVectorTest.
                                          SupportVectorRegressionTest))
        _suite.addTest(unittest.makeSuite(unittests.RemoveInactive.
                                          RemoveInactive))
    """

    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    test_suite = suite()
    runner.run(suite())
