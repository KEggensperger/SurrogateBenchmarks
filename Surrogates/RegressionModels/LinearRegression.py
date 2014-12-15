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


class LinearRegression(ScikitBaseClass.ScikitBaseClass):

    def __init__(self, sp, encode, rng, **kwargs):
        ScikitBaseClass.ScikitBaseClass.__init__(self, sp=sp, encode=encode,
                                                 rng=rng, **kwargs)
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
            lr = sklearn.linear_model.LinearRegression(fit_intercept=True,
                                                       normalize=False,
                                                       copy_X=True)
            lr.fit(scaled_x, y)
            self._model = lr
        except Exception, e:
            print "Training failed", e.message
            lr = None
        duration = time.time() - start
        self._training_finished = True
        return duration