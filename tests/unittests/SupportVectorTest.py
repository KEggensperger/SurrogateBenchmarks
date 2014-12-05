
import numpy
import os
import unittest

from sklearn.metrics import mean_squared_error

import Surrogates.RegressionModels.SupportVectorRegression
import Surrogates.DataExtraction.pcs_parser as pcs_parser
from Surrogates.DataExtraction.data_util import read_csv


class SupportVectorRegressionTest(unittest.TestCase):
    _checkpoint = None
    _data = None
    _para_header = None
    _sp = None

    def setUp(self):
        self._sp = pcs_parser.read(file(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Testdata/nips2011.pcs")))
        # Read data from csv
        header, self._data = read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Testdata/hpnnet_nocv_convex_all_fastrf_results.csv"),
                                      has_header=True, num_header_rows=3)
        self._para_header = header[0][:-2]
        self._checkpoint = hash(numpy.array_repr(self._data))

    def tearDown(self):
        self.assertEqual(hash(numpy.array_repr(self._data)), self._checkpoint)

    def test_train(self):
        model = Surrogates.RegressionModels.SupportVectorRegression.SupportVectorRegression(sp=self._sp, encode=False,
                                                                                            rng=1, debug=True)
        x_train_data = self._data[:1000, :-2]
        y_train_data = self._data[:1000, -1]
        x_test_data = self._data[1000:, :-2]
        y_test_data = self._data[1000:, -1]

        model.train(x=x_train_data, y=y_train_data, param_names=self._para_header)

        should_be_lower = [None, -29.6210089736, 0.201346561323, 0, -20.6929600285, 0, 0, 0, 4.60517018599, 0,
                           2.77258872224, 0, 0, 0.502038871605, -17.2269829469]
        should_be_upper = [None, -7.33342451433, 1.99996215592, 1, -6.92778489957, 2, 1, 1, 9.20883924585, 1,
                           6.9314718056, 3, 1, 0.998243871085, 4.72337617503]

        for idx in range(x_train_data.shape[1]):
            self.assertEqual(model._scale_info[0][idx], should_be_lower[idx])
            self.assertEqual(model._scale_info[1][idx], should_be_upper[idx])

        y = model.predict(x=x_train_data[1])
        print "Is: %100.70f, Should: %f" % (y, y_train_data[1])
        self.assertAlmostEqual(y[0], 0.37112789312415417)

        print "Predict whole data"
        y_whole = model.predict(x=x_test_data)
        mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
        print "MSE: %100.70f" % mse
        self.assertAlmostEqual(mse, 0.00754033374998010381962121329024739679880440235137939453125)

        # Try the same with encoded features
        model = Surrogates.RegressionModels.SupportVectorRegression.SupportVectorRegression(sp=self._sp, rng=1,
                                                                                            encode=True, debug=True)
        #print data[:10, :-2]
        model.train(x=x_train_data, y=y_train_data, param_names=self._para_header, rng=1)

        y = model.predict(x=self._data[1, :-2])
        print "Is: %100.70f, Should: %f" % (y, self._data[1, -2])
        self.assertAlmostEqual(y[0], 0.3661722483936813432592316530644893646240234375)

        print "Predict whole data"
        y_whole = model.predict(x=x_test_data)
        mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
        print "MSE: %100.70f" % mse
        self.assertAlmostEqual(mse, 0.00817690485947193158866586060184999951161444187164306640625)
