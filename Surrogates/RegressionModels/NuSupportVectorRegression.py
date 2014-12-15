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

from scipy.stats import uniform

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterSampler
from sklearn.svm import NuSVR

class NuSupportVectorRegression(ScikitBaseClass.ScikitBaseClass):

    def __init__(self, sp, encode, rng, **kwargs):
        ScikitBaseClass.ScikitBaseClass.__init__(self, sp=sp, encode=encode,
                                                 rng=rng, **kwargs)
        self._name = "NuSVR " + str(encode)
        self._max_number_train_data = 2000

    def _random_search(self, random_iter, x, y, kernel_cache_size):
        # Default Values
        c = 1.0
        gamma = 0.0
        nu = 0.5
        best_score = -sys.maxint

        if random_iter > 0:
            sys.stdout.write("Do a random search %d times" % random_iter)
            param_dist = {"C": numpy.power(2.0, range(-5, 16)),
                          "gamma": numpy.power(2.0, range(-15, 4)),
                          "nu": uniform(loc=0.0001, scale=1-0.0001)}
            param_list = [{"C": c, "gamma": gamma, "nu": nu}, ]
            param_list.extend(list(ParameterSampler(param_dist,
                                                    n_iter=random_iter-1,
                                                    random_state=self._rng)))
            for idx, d in enumerate(param_list):
                nusvr = NuSVR(kernel='rbf',
                              gamma=d['gamma'],
                              C=d['C'],
                              nu=d['nu'],
                              random_state=self._rng,
                              cache_size=kernel_cache_size)
                train_x, test_x, train_y, test_y = \
                    train_test_split(x, y, test_size=0.5, random_state=self._rng)
                self._check_scaling(scaled_x=train_x)
                nusvr.fit(train_x, train_y)
                sc = nusvr.score(test_x, test_y)
                # Tiny output
                m = "."
                if idx % 10 == 0:
                    m = "#"
                if sc > best_score:
                    m = "<"
                    best_score = sc
                    c = d['C']
                    gamma = d['gamma']
                    nu = d['nu']
                sys.stdout.write(m)
                sys.stdout.flush()
            sys.stdout.write("Using C: %f, nu: %f and Gamma: %f\n" %
                             (c, nu, gamma))
        return nu, c, gamma

    def train(self, x, y, param_names, random_search=100,
              kernel_cache_size=2000, **kwargs):
        if self._debug:
            print "Before preprocessing: 1st sample:\n", x[0]
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
        nu, c, gamma = self._random_search(random_iter=100, x=scaled_x, y=y,
                                           kernel_cache_size=kernel_cache_size)

        # Now train model
        try:
            nusvr = NuSVR(gamma=gamma, C=c, nu=nu, random_state=self._rng,
                          cache_size=kernel_cache_size)
            nusvr.fit(scaled_x, y)
            self._model = nusvr
        except Exception, e:
            print "Training failed", e.message
            svr = None
        duration = time.time() - start
        self._training_finished = True
        return duration