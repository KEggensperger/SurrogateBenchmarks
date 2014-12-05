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

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import ParameterSampler
from sklearn.cross_validation import train_test_split

from Surrogates.DataExtraction.data_util import read_csv


class GradientBoosting(ScikitBaseClass.ScikitBaseClass):

    def __init__(self, sp, encode, rng=1, **kwargs):
        ScikitBaseClass.ScikitBaseClass.__init__(self, sp=sp, encode=encode, rng=rng, **kwargs)
        self._name = "GradientBoosting " + str(encode)

    def _random_search(self, random_iter, x, y):
        # Default values
        max_features = x.shape[1]
        learning_rate = 0.1   # [0.1, 0.05, 0.02, 0.01],
        max_depth = 3         # [4, 6],
        min_samples_leaf = 1  # [3, 5, 9, 17],
        n_estimators = 100    #
        best_score = -sys.maxint

        if random_iter > 0:
            sys.stdout.write("Do a random search %d times" % random_iter)

            param_dist = {"max_features": numpy.linspace(0.1, 1, num=10),
                          "learning_rate": 2**numpy.linspace(-1, -10, num=10),
                          "max_depth": range(1, 11),
                          "min_samples_leaf": range(2, 20, 2),
                          "n_estimators": range(10, 110, 10)}
            param_list = [{"max_features": max_features,
                           "learning_rate": learning_rate,
                           "max_depth": max_depth,
                           "min_samples_leaf": min_samples_leaf,
                           "n_estimators": n_estimators}]
            param_list.extend(list(ParameterSampler(param_dist, n_iter=random_iter-1, random_state=self._rng)))

        for idx, d in enumerate(param_list):
            gb = GradientBoostingRegressor(loss='ls',
                                           learning_rate=d["learning_rate"],
                                           n_estimators=d["n_estimators"],
                                           subsample=1.0,
                                           min_samples_split=2,
                                           min_samples_leaf=d["min_samples_leaf"],
                                           max_depth=d["max_depth"],
                                           init=None,
                                           random_state=self._rng,
                                           max_features=d["max_features"],
                                           alpha=0.9,
                                           verbose=0)
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.5, random_state=self._rng)
            gb.fit(train_x, train_y)
            sc = gb.score(test_x, test_y)
            # Tiny output
            m = "."
            if idx % 10 == 0:
                m = "#"
            if sc > best_score:
                m = "<"
                best_score = sc
                max_features = d["max_features"]
                learning_rate = d["learning_rate"]
                max_depth = d["max_depth"]
                min_samples_leaf = d["min_samples_leaf"]
                n_estimators = d["n_estimators"]
            sys.stdout.write(m)
            sys.stdout.flush()
        sys.stdout.write("Using max_features %f, learning_rate: %f, max_depth: %d, min_samples_leaf: %d, "
                         "and n_estimators: %d\n" %
                         (max_features, learning_rate, max_depth, min_samples_leaf, n_estimators))

        return max_features, learning_rate, max_depth, min_samples_leaf, n_estimators

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

        # Do a random search
        max_features, learning_rate, max_depth, min_samples_leaf, n_estimators = self._random_search(random_iter=100,
                                                                                                     x=scaled_x, y=y)
        # Now train model
        gb = GradientBoostingRegressor(loss='ls',
                                       learning_rate=learning_rate,
                                       n_estimators=n_estimators,
                                       subsample=1.0,
                                       min_samples_split=2,
                                       min_samples_leaf=min_samples_leaf,
                                       max_depth=max_depth,
                                       init=None,
                                       random_state=self._rng,
                                       max_features=max_features,
                                       alpha=0.9,
                                       verbose=0)
        gb.fit(scaled_x, y)
        self._model = gb

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

    model = GradientBoosting(sp=sp, encode=False, debug=True)
    x_train_data = data[:1000, :-2]
    y_train_data = data[:1000, -1]
    x_test_data = data[1000:, :-2]
    y_test_data = data[1000:, -1]

    model.train(x=x_train_data, y=y_train_data, param_names=para_header, rng=1)

    y = model.predict(x=x_train_data[1, :])
    print "Is: %100.70f, Should: %f" % (y, y_train_data[1])
    assert y[0] == 0.45366000254662230961599789225147105753421783447265625

    print "Predict whole data"
    y_whole = model.predict(x=x_test_data)
    mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
    print "MSE: %100.70f" % mse
    assert mse == 0.00188246958253847243396073007914992558653466403484344482421875

    print "Soweit so gut"

    # Try the same with encoded features
    model = GradientBoosting(sp=sp, encode=True, debug=True)
    #print data[:10, :-2]
    model.train(x=x_train_data, y=y_train_data, param_names=para_header, rng=1)

    y = model.predict(x=x_train_data[1, :])
    print "Is: %100.70f, Should: %f" % (y, y_train_data[1])
    assert y[0] == 0.460818965082699205648708584703854285180568695068359375

    print "Predict whole data"
    y_whole = model.predict(x=x_test_data)
    mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
    print "MSE: %100.70f" % mse
    assert mse == 0.002064362783199560034963493393433964229188859462738037109375

    assert hash(numpy.array_repr(data)) == checkpoint

if __name__ == "__main__":
    outer_start = time.time()
    test()
    dur = time.time() - outer_start
    print "TESTING TOOK: %f sec" % dur
