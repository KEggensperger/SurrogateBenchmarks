
import cPickle
import os
import unittest

import numpy
from sklearn.metrics import mean_squared_error

import Surrogates.RegressionModels.LinearRegression
import Surrogates.DataExtraction.pcs_parser as pcs_parser
from Surrogates.DataExtraction.data_util import read_csv


class LinearRegressionTest(unittest.TestCase):
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
        model = Surrogates.RegressionModels.LinearRegression.LinearRegression(sp=self._sp, encode=False, rng=1, debug=True)
        x_train_data = self._data[:1000, :-2]
        y_train_data = self._data[:1000, -1]
        x_test_data = self._data[1000:, :-2]
        y_test_data = self._data[1000:, -1]

        model.train(x=x_train_data, y=y_train_data, param_names=self._para_header, rng=1)

        str_data = x_train_data[1, :]
        str_data = list(str_data)
        y1 = model.predict(x=str_data)
        #print "Is: %100.70f, Should: %f" % (y1, y_train_data[1])
        str_data[0] = str(str_data[0])
        y2 = model.predict(x=str_data)
        #print "Is: %100.70f, Should: %f" % (y2, y_train_data[1])
        self.assertEqual(y1, y2)

        y = model.predict(x=x_train_data[1])
        #print "Is: %100.70f, Should: %f" % (y, y_train_data[1])
        self.assertAlmostEqual(y[0], 0.337924630401289560754918284)
        self.assertEqual(y, y1)

        #print "Predict whole data"
        y_whole = model.predict(x=x_test_data)
        mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
        #print "MSE: %100.70f" % mse
        self.assertAlmostEqual(mse, 0.009198984490357622889611377)

        # Try the same with encoded features
        model = Surrogates.RegressionModels.LinearRegression.\
            LinearRegression(sp=self._sp, rng=1, encode=True, debug=True)
        #print data[:10, :-2]
        model.train(x=x_train_data, y=y_train_data,
                    param_names=self._para_header, rng=1)

        y = model.predict(x=self._data[1, :-2])
        #print "Is: %100.70f, Should: %f" % (y, self._data[1, -2])
        self.assertAlmostEqual(y[0], 0.3359375)

        #print "Predict whole data"
        y_whole = model.predict(x=x_test_data)
        mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
        #print "MSE: %100.70f" % mse
        self.assertAlmostEqual(mse, 0.0091497876001024742)

        fn = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "Testdata/testLinear.pkl")
        fh = open(fn, "wb")
        cPickle.dump(model, fh)
        fh.close()
        a = cPickle.load(file(fn))

        #print "Predict whole data"
        y_whole = a.predict(x=x_test_data)
        mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
        #print "MSE: %100.70f" % mse
        self.assertAlmostEqual(mse, 0.0091497876001024742)
        self.assertEqual(a._name, "Linear_regression True")
        os.remove(fn)
