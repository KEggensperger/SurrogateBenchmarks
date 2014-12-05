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
from sklearn.ensemble import RandomForestRegressor

from Surrogates.DataExtraction.data_util import read_csv


class RandomForest(ScikitBaseClass.ScikitBaseClass):

    def __init__(self, sp, encode, rng, **kwargs):
        ScikitBaseClass.ScikitBaseClass.__init__(self, sp=sp, encode=encode, rng=rng, **kwargs)
        self._name = "RandomForest " + str(encode)

    def _random_search(self, random_iter, x, y):
        # Default Values
        n_estimators = 10
        min_samples_split = 2
        max_features = x.shape[1]
        best_score = -sys.maxint

        if random_iter > 0:
            sys.stdout.write("Do a random search %d times" % random_iter)
            param_dist = {"n_estimators": range(10, 110, 10),
                          "min_samples_split": range(2, 20, 2),
                          "max_features":  numpy.linspace(0.1, 1, num=10)}
            param_list = [{"n_estimators": n_estimators,
                           "min_samples_split": min_samples_split,
                           "max_features": max_features}, ]
            param_list.extend(list(ParameterSampler(param_dist, n_iter=random_iter-1, random_state=self._rng)))
            for idx, d in enumerate(param_list):
                rf = RandomForestRegressor(n_estimators=int(d["n_estimators"]),
                                           criterion='mse',
                                           max_depth=None,
                                           min_samples_split=int(d["min_samples_split"]),
                                           min_samples_leaf=1,
                                           max_features=d["max_features"],
                                           max_leaf_nodes=None,
                                           bootstrap=True,
                                           oob_score=False,
                                           n_jobs=1,
                                           random_state=self._rng,
                                           verbose=0, min_density=None, compute_importances=None)

                train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.5, random_state=self._rng)
                rf.fit(train_x, train_y)
                sc = rf.score(test_x, test_y)
                # Tiny output
                m = "."
                if idx % 10 == 0:
                    m = "#"
                if sc > best_score:
                    m = "<"
                    best_score = sc
                    n_estimators = int(d["n_estimators"])
                    min_samples_split = int(d["min_samples_split"])
                    max_features = d["max_features"]
                sys.stdout.write(m)
                sys.stdout.flush()
            sys.stdout.write("Using n_estimators: %d, min_samples_split: %d, and max_features: %f\n" % (n_estimators, min_samples_split, max_features))
        return n_estimators, min_samples_split, max_features

    def train(self, x, y, param_names, random_search=100, **kwargs):
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
        n_estimators, min_samples_split, max_features = self._random_search(random_iter=100, x=scaled_x, y=y)

        rf = RandomForestRegressor(n_estimators=n_estimators,
                                   criterion='mse',
                                   max_depth=None,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=1,
                                   max_features=max_features,
                                   max_leaf_nodes=None,
                                   bootstrap=True,
                                   oob_score=False,
                                   n_jobs=1,
                                   random_state=self._rng,
                                   verbose=0, min_density=None, compute_importances=None)
        rf.fit(scaled_x, y)
        self._model = rf

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

    model = RandomForest(sp=sp, encode=False, debug=True, rng=1)
    x_train_data = data[:1000, :-2]
    y_train_data = data[:1000, -1]
    x_test_data = data[1000:, :-2]
    y_test_data = data[1000:, -1]

    model.train(x=x_train_data, y=y_train_data, param_names=para_header)
    assert model._model.get_params()['n_estimators'] == 70
    assert model._model.get_params()['min_samples_split'] == 2
    assert model._model.get_params()['max_features'] == 0.4

    str_data = x_train_data[1, :]
    str_data = list(str_data)
    y1 = model.predict(x=str_data)
    print "Is: %100.70f, Should: %f" % (y1, y_train_data[1])
    str_data[0] = str(str_data[0])
    y2 = model.predict(x=str_data)
    print "Is: %100.70f, Should: %f" % (y2, y_train_data[1])
    assert y1 == y2

    y = model.predict(x=x_train_data[1, :])
    print "Is: %100.70f, Should: %f" % (y, y_train_data[1])
    assert y[0] == 0.438562527380952638100808371746097691357135772705078125

    print "Predict whole data"
    y_whole = model.predict(x=x_test_data)
    mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
    print "MSE: %100.70f" % mse
    assert mse == 0.0027243939040595683935552795418288951623253524303436279296875

    print "Soweit so gut"

    # Try the same with encoded features
    model = RandomForest(sp=sp, encode=True, debug=True, rng=1)
    #print data[:10, :-2]
    model.train(x=x_train_data, y=y_train_data, param_names=para_header)

    y = model.predict(x=x_train_data[1, :])
    print "Is: %100.70f, Should: %f" % (y, y_train_data[1])
    assert y[0] == 0.435386310714285962308878197291051037609577178955078125

    print "Predict whole data"
    y_whole = model.predict(x=x_test_data)
    mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
    print "MSE: %100.70f" % mse
    assert mse == 0.0025204000192945203058514369587328474153764545917510986328125

    assert hash(numpy.array_repr(data)) == checkpoint

if __name__ == "__main__":
    outer_start = time.time()
    test()
    dur = time.time() - outer_start
    print "TESTING TOOK: %f sec" % dur