##
# SurrogateBenchmarks: A program making it easy to benchmark hyperparameter
# optimization software .
# Copyright (C) 2014 Katharina Eggensperger
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__author__ = ["Katharina Eggensperger"]

import time

import numpy
numpy.random.seed(1)

import sklearn.linear_model
from Surrogates.RegressionModels import ScikitBaseClass

from Surrogates.DataExtraction.data_util import read_csv


class LinearRegression(ScikitBaseClass.ScikitBaseClass):

    def __init__(self, sp, encode, rng, **kwargs):
        ScikitBaseClass.ScikitBaseClass.__init__(self, sp=sp, encode=encode, rng=rng, **kwargs)
        self._name = "Linear_regression " + str(encode)

    def train(self, x, y, param_names, **kwargs):
        start = time.time()
        scaled_x = self._set_and_preprocess(x=x, param_names=param_names)

        # Check that each input is between 0 and 1
        self._check_scaling(scaled_x=scaled_x)

        if self._debug:
            print "Shape of training data: ", scaled_x.shape
            print "Param names: ", self._used_param_names
            print "First training sample\n", scaled_x[0]
            print "Encode: ", self._encode

        # Now train model
        start = time.time()
        try:
            lr = sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
            lr.fit(scaled_x, y)
            self._model = lr
        except Exception, e:
            print "Training failed", e.message
            lr = None
        duration = time.time() - start
        self._training_finished = True
        return duration


def test():
    from sklearn.metrics import mean_squared_error
    import Surrogates.DataExtraction.pcs_parser as pcs_parser
    sp = pcs_parser.read(file("/home/eggenspk/Surrogates/Data_extraction/Experiments2014/hpnnet/smac_2_06_01-dev/nips2011.pcs"))
    # Read data from csv
    header, data = read_csv("/home/eggenspk/Surrogates/Data_extraction/hpnnet_nocv_convex_all/hpnnet_nocv_convex_all_fastrf_results.csv",
                            has_header=True, num_header_rows=3)
    para_header = header[0][:-2]
    type_header = header[1]
    cond_header = header[2]
    #print data.shape
    checkpoint = hash(numpy.array_repr(data))
    assert checkpoint == 246450380584980815

    model = LinearRegression(sp=sp, encode=False, debug=True, rng=1)
    x_train_data = data[:1000, :-2]
    y_train_data = data[:1000, -1]
    x_test_data = data[1000:, :-2]
    y_test_data = data[1000:, -1]

    model.train(x=x_train_data, y=y_train_data, param_names=para_header)

    str_data = x_train_data[1, :]
    str_data = list(str_data)
    y1 = model.predict(x=str_data)
    print "Is: %100.70f, Should: %f" % (y1, y_train_data[1])
    str_data[0] = str(str_data[0])
    y2 = model.predict(x=str_data)
    print "Is: %100.70f, Should: %f" % (y2, y_train_data[1])
    assert y1 == y2

    y = model.predict(x=data[1, :-2])
    print "Is: %100.70f, Should: %f" % (y, data[1, -2])
    assert y[0] == 0.337924630401289560754918284146697260439395904541015625

    print "Predict whole data"
    y_whole = model.predict(x=x_test_data)
    mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
    print "MSE: %100.70f" % mse
    assert mse == 0.00919898449035762288961137755904928781092166900634765625

    print "Soweit so gut"

    # Try the same with encoded features
    model = LinearRegression(sp=sp, encode=True, debug=True, rng=1)
    #print data[:10, :-2]
    model.train(x=x_train_data, y=y_train_data, param_names=para_header)

    y = model.predict(x=data[1, :-2])
    print "Is: %100.70f, Should: %f" % (y, data[1, -2])
    assert y[0] == 0.3359375

    print "Predict whole data"
    y_whole = model.predict(x=x_test_data)
    mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
    print "MSE: %100.70f" % mse
    assert mse == 0.00914978760010247416101236694885301403701305389404296875

    assert hash(numpy.array_repr(data)) == checkpoint

if __name__ == "__main__":
    outer_start = time.time()
    test()
    dur = time.time() - outer_start
    print "TESTING TOOK: %f sec" % dur
