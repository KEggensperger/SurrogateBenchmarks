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

from Surrogates.RegressionModels import ScikitBaseClass

import sys
import time

import numpy
numpy.random.seed(1)

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterSampler
from sklearn.svm import SVR

from Surrogates.DataExtraction.data_util import read_csv


class SupportVectorRegression(ScikitBaseClass.ScikitBaseClass):

    def __init__(self, sp, encode, rng, **kwargs):
        ScikitBaseClass.ScikitBaseClass.__init__(self, sp=sp, encode=encode, rng=rng, **kwargs)
        self._name = "SVR " + str(encode)
        #self._max_number_train_data = 2000

    def _random_search(self, random_iter, x, y, kernel_cache_size):
        # Default Values
        c = 1.0
        gamma = 0.0
        best_score = -sys.maxint

        if random_iter > 0:
            sys.stdout.write("Do a random search %d times" % random_iter)
            param_dist = {"C": numpy.power(2.0, range(-5, 16)),
                          "gamma": numpy.power(2.0, range(-15, 4))}
            param_list = [{"C": c, "gamma": gamma}, ]
            param_list.extend(list(ParameterSampler(param_dist, n_iter=random_iter-1, random_state=self._rng)))
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.5, random_state=self._rng)

            for idx, d in enumerate(param_list):
                svr = SVR(kernel='rbf',
                          gamma=d['gamma'],
                          C=d['C'],
                          random_state=self._rng,
                          cache_size=kernel_cache_size)
                svr.fit(train_x, train_y)
                sc = svr.score(test_x, test_y)
                # Tiny output
                m = "."
                if idx % 10 == 0:
                    m = "#"
                if sc > best_score:
                    m = "<"
                    best_score = sc
                    c = d['C']
                    gamma = d['gamma']
                sys.stdout.write(m)
                sys.stdout.flush()
            sys.stdout.write("Using C: %f and Gamma: %f\n" % (c, gamma))
        return c, gamma

    def train(self, x, y, param_names, random_search=100, kernel_cache_size=2000, **kwargs):
        if self._debug:
            print "First training sample\n", x[0]
        start = time.time()
        scaled_x = self._set_and_preprocess(x=x, param_names=param_names)


        # Check that each input is between 0 and 1
        self._check_scaling(scaled_x=scaled_x)

        if self._debug:
            print "Shape of training data: ", scaled_x.shape
            print "Param names: ", self._used_param_names
            print "First training sample\n", scaled_x[0]
            print "Encode: ", self._encode

        # Do a random search
        c, gamma = self._random_search(random_iter=100, x=scaled_x, y=y, kernel_cache_size=kernel_cache_size)

        # Now train model
        try:
            svr = SVR(gamma=gamma, C=c, random_state=self._rng, cache_size=kernel_cache_size)
            svr.fit(scaled_x, y)
            self._model = svr
        except Exception, e:
            print "Training failed", e.message
            svr = None
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
    para_header = header[0]
    type_header = header[1]
    cond_header = header[2]
    checkpoint = hash(numpy.array_repr(data))
    assert checkpoint == 246450380584980815

    model = SupportVectorRegression(sp=sp, encode=False, debug=True, rng=1)
    x_train_data = data[:1000, :-2]
    y_train_data = data[:1000, -1]
    x_test_data = data[1000:, :-2]
    y_test_data = data[1000:, -1]

    model.train(x=x_train_data, y=y_train_data, param_names=para_header)
    print model._scale_info
    assert model._model.get_params()['gamma'] == 0.015625, "%100.20f" % model._model.get_params()['gamma']
    assert model._model.get_params()['C'] == 256, "%100.20f" % model._model.get_params()['C']
    print model._model.get_params()

    y = model.predict(x=x_train_data[1, :])
    print "Is: %100.70f, Should: %f" % (y, y_train_data[1])
    assert y[0] == 0.371127893124154173420947699923999607563018798828125

    print "Predict whole data"
    y_whole = model.predict(x=x_test_data)
    mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
    print "MSE: %100.70f" % mse
    assert mse == 0.00754033374998010381962121329024739679880440235137939453125

    print "Soweit so gut"

    # Try the same with encoded features
    model = SupportVectorRegression(sp=sp, encode=True, debug=True, rng=1)
    #print data[:10, :-2]
    model.train(x=x_train_data, y=y_train_data, param_names=para_header)

    y = model.predict(x=x_train_data[1, :])
    print "Is: %100.70f, Should: %f" % (y, y_train_data[1])
    assert y[0] == 0.3661722483936813432592316530644893646240234375

    print "Predict whole data"
    y_whole = model.predict(x=x_test_data)
    mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
    print "MSE: %100.70f" % mse
    assert mse == 0.00817690485947193158866586060184999951161444187164306640625

    assert hash(numpy.array_repr(data)) == checkpoint

if __name__ == "__main__":
    outer_start = time.time()
    test()
    dur = time.time() - outer_start
    print "TESTING TOOK: %f sec" % dur
