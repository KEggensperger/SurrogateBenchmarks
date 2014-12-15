
import cPickle
import os
import unittest

import numpy
numpy.random.seed(1)
from sklearn.metrics import mean_squared_error

import Surrogates.RegressionModels.GaussianProcess
import Surrogates.DataExtraction.pcs_parser as pcs_parser
from Surrogates.DataExtraction.data_util import read_csv


class GaussianProcessTest(unittest.TestCase):
    _checkpoint = None
    _data = None
    _para_header = None
    _sp = None

    def setUp(self):
        self._sp = pcs_parser.read(file(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "Testdata/nips2011.pcs")))
        # Read data from csv
        header, self._data = read_csv(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "Testdata/hpnnet_nocv_convex_all_fastrf_results.csv"),
            has_header=True, num_header_rows=3)
        self._para_header = header[0][:-2]
        self._checkpoint = hash(numpy.array_repr(self._data))
        self.assertEqual(246450380584980815, self._checkpoint)

    def tearDown(self):
        self.assertEqual(hash(numpy.array_repr(self._data)), self._checkpoint)

    def test_train(self):
        x_train_data = self._data[:100, :-2]
        y_train_data = self._data[:100, -1]
        x_test_data = self._data[100:, :-2]
        y_test_data = self._data[100:, -1]

        model = Surrogates.RegressionModels.GaussianProcess.\
            GaussianProcess(sp=self._sp, rng=1, encode=False, debug=True)

        model.train(x=x_train_data, y=y_train_data,
                    param_names=self._para_header)

        y = model.predict(x=x_train_data[1, :])
        print "Is: %100.70f, Should: %f" % (y, y_train_data[1])
        self.assertAlmostEqual(y[0], 0.47074515351490015)

        print "Predict whole data"
        y_whole = model.predict(x=x_test_data)
        mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
        print "MSE: %100.70f" % mse
        self.assertAlmostEqual(mse, 0.0062575986090041905)

        print "Soweit so gut"

        # Try the same with encoded features
        model = Surrogates.RegressionModels.GaussianProcess.\
            GaussianProcess(sp=self._sp, rng=1, encode=True, debug=True)
        #print data[:10, :-2]
        model.train(x=x_train_data, y=y_train_data,
                    param_names=self._para_header, rng=1)

        y = model.predict(x=x_train_data[1, :])
        print "Is: %100.70f, Should: %f" % (y, y_train_data[1])
        self.assertAlmostEqual(y[0], 0.46467166529432441)

        print "Predict whole data"
        y_whole = model.predict(x=x_test_data)
        mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
        print "MSE: %100.70f" % mse
        self.assertAlmostEqual(mse, 0.0091926512804233057)

        fn = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "Testdata/testGP.pkl")
        fh = open(fn, "wb")
        cPickle.dump(model, fh)
        fh.close()
        a = cPickle.load(file(fn))

        print "Predict whole data"
        y_whole = a.predict(x=x_test_data)
        mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
        print "MSE: %100.70f" % mse
        self.assertAlmostEqual(mse, 0.0091926512804233057)
        self.assertEqual(a._name, "GP True")
        self.assertEqual(a._mcmc_iters, 10)

        os.remove(fn)
