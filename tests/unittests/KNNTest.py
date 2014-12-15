import cPickle
import numpy
numpy.random.seed(1)

import os
import unittest

from sklearn.metrics import mean_squared_error

import Surrogates.RegressionModels.KNN
import Surrogates.DataExtraction.pcs_parser as pcs_parser
from Surrogates.DataExtraction.data_util import read_csv


class KNNTest(unittest.TestCase):
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
        model = Surrogates.RegressionModels.KNN.KNN(sp=self._sp, encode=False,
                                                    rng=1, debug=True)
        x_train_data = self._data[:1000, :-2]
        y_train_data = self._data[:1000, -1]
        x_test_data = self._data[1000:, :-2]
        y_test_data = self._data[1000:, -1]

        self.assertEqual(hash(numpy.array_repr(x_train_data)),
                         -4233919799601849470)
        self.assertEqual(hash(numpy.array_repr(y_train_data)),
                         -5203961977442829493)

        model.train(x=x_train_data, y=y_train_data, param_names=self._para_header)

        lower, upper = model._scale_info
        should_be_lower = [None, -29.6210089736, 0.201346561323, 0,
                           -20.6929600285, 0, 0, 0, 4.60517018599, 0,
                           2.77258872224, 0, 0, 0.502038871605, -17.2269829469]
        should_be_upper = [None, -7.33342451433, 1.99996215592, 1,
                           -6.92778489957, 2, 1, 1, 9.20883924585, 1,
                           6.9314718056, 3, 1, 0.998243871085, 4.72337617503]

        for idx in range(x_train_data.shape[1]):
            self.assertEqual(lower[idx], should_be_lower[idx])
            self.assertEqual(upper[idx], should_be_upper[idx])

        y = model.predict(x=x_train_data[1, :])
        print "Is: %100.70f, Should: %f" % (y, y_train_data[1])
        self.assertAlmostEqual(y[0], 0.281714285714)

        print "Predict whole data"
        y_whole = model.predict(x=x_test_data)
        mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
        print "MSE: %100.70f" % mse
        self.assertAlmostEqual(mse, 0.0093528907735581299)

        # Try the same with encoded features
        model = Surrogates.RegressionModels.KNN.KNN(sp=self._sp, encode=True,
                                                    rng=1, debug=True)
        #print data[:10, :-2]
        model.train(x=x_train_data, y=y_train_data,
                    param_names=self._para_header, rng=1)

        y = model.predict(x=self._data[1, :-2])
        print "Is: %100.70f, Should: %f" % (y, self._data[1, -2])
        self.assertAlmostEqual(y[0], 0.28522216666666667)

        print "Predict whole data"
        y_whole = model.predict(x=x_test_data)
        mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
        print "MSE: %100.70f" % mse
        self.assertAlmostEqual(mse, 0.0092382390066771715)

        fn = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "Testdata/testknn.pkl")
        fh = open(fn, "wb")
        cPickle.dump(model, fh)
        fh.close()
        a = cPickle.load(file(fn))

        #print "Predict whole data"
        y_whole = a.predict(x=x_test_data)
        mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
        #print "MSE: %100.70f" % mse
        self.assertAlmostEqual(mse, 0.0092382390066771715)
        self.assertEqual(a._name, "KNN True")
        os.remove(fn)


"""
def test():
    from sklearn.metrics import mean_squared_error
    import Surrogates.DataExtraction.pcs_parser as pcs_parser
    sp = pcs_parser.read(file("/home/eggenspk/Surrogates/Data_extraction/Experiments2014/hpnnet/smac_2_06_01-dev/nips2011.pcs"))
    # Read data from csv
    header, data = read_csv("/home/eggenspk/Surrogates/Data_extraction/hpnnet_nocv_convex_all/hpnnet_nocv_convex_all_fastrf_results.csv",
                            has_header=True, num_header_rows=3)
    para_header = header[0]
    type_header = header[1]
    cond_header = header[2]
    checkpoint = hash(numpy.array_repr(data))
    assert checkpoint == 246450380584980815

    model = KNN(sp=sp, encode=False, debug=True)
    x_train_data = data[:1000, :-2]
    y_train_data = data[:1000, -1]
    x_test_data = data[1000:, :-2]
    y_test_data = data[1000:, -1]

    model.train(x=x_train_data, y=y_train_data, param_names=para_header, rng=1)

    y = model.predict(x=x_train_data[1, :])
    print "Is: %100.70f, Should: %f" % (y, y_train_data[1])
    assert y[0] == 0.28171428571428569487267168369726277887821197509765625

    print "Predict whole data"
    y_whole = model.predict(x=x_test_data)
    mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
    print "MSE: %100.70f" % mse
    assert mse == 0.009352890773558129866582788736195652745664119720458984375

    print "Soweit so gut"

    # Try the same with encoded features
    model = KNN(sp=sp, encode=True, debug=True)
    #print data[:10, :-2]
    model.train(x=x_train_data, y=y_train_data, param_names=para_header, rng=1)

    y = model.predict(x=x_train_data[1, :])
    print "Is: %100.70f, Should: %f" % (y, y_train_data[1])
    assert y[0] == 0.28522216666666666551321895894943736493587493896484375

    print "Predict whole data"
    y_whole = model.predict(x=x_test_data)
    mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
    print "MSE: %100.70f" % mse
    assert mse == 0.0092382390066771714887128297277740784920752048492431640625

    assert hash(numpy.array_repr(data)) == checkpoint

if __name__ == "__main__":
    outer_start = time.time()
    test()
    dur = time.time() - outer_start
    print "TESTING TOOK: %f sec" % dur
"""