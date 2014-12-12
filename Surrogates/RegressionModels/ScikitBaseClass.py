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

import copy

import numpy

import model_util
import scaler
import RegressionBaseClass


class ScikitBaseClass(RegressionBaseClass.RegressionBaseClass):

    def __init__(self, sp, encode, rng, debug=False, **kwargs):
        numpy.random.seed(rng)
        RegressionBaseClass.RegressionBaseClass.__init__(self, sp=sp, rng=rng,
                                                         encode=encode,
                                                         debug=debug)

    def _set_and_preprocess(self, x, param_names):
        assert not self._training_finished, "Training is not finished"
        scaled_x = copy.deepcopy(x)

        #if self._scale_info is None:
        #    raise ValueError("No scale info available for this model")
        self.set_number_of_folds(scaled_x)
        self._param_names = param_names

        # 1. Replace strings with numbers
        scaled_x = model_util.replace_cat_variables(scaled_x, self._catdict,
                                                    self._param_names)

        # 2. Maybe encode data
        if self._encode:
            scaled_x, self._used_param_names = \
                model_util.encode(sp=self._sp, x=scaled_x,
                                  param_names=self._param_names,
                                  num_folds=self._num_folds,
                                  catdict=self._catdict)
        else:
            self._used_param_names = self._param_names

        # 4. Set statistics
        if self._encode:
            #raise NotImplementedError("One-hot-encoding is not implemented")
            self._scale_info = \
                scaler.get_x_info_scaling_all(sp=self._sp, x=scaled_x,
                                              param_names=self._used_param_names,
                                              num_folds=self._num_folds,
                                              encoded=self._encode)
        else:
            self._scale_info = \
                scaler.get_x_info_scaling_all(sp=self._sp, x=scaled_x,
                                              param_names=self._used_param_names,
                                              num_folds=self._num_folds,
                                              encoded=self._encode)

        # 4. Scale data
        scaled_x = scaler.scale(scale_info=self._scale_info, x=scaled_x)
        return scaled_x

    def _preprocess(self, x):
        assert self._training_finished, "Training is not finished"
        x_scaled = x
        # 1. Relace strings with numbers
        x_scaled = model_util.replace_cat_variables(x=x_scaled,
                                                    catdict=self._catdict,
                                                    param_names=self._param_names)

        # 2. Maybe encode data
        if self._encode:
            x_scaled, tmp_param_names = \
                model_util.encode(sp=self._sp, x=x_scaled,
                                  param_names=self._param_names,
                                  num_folds=self._num_folds,
                                  catdict=self._catdict)
            assert self._used_param_names == tmp_param_names
        # 3. Scale data
        x_scaled = scaler.scale(scale_info=self._scale_info, x=x_scaled)
        return x_scaled

    def _check_scaling(self, scaled_x, tol=0):
        for idx, p in enumerate(self._used_param_names):
            if p == "duration" or p == "performance":
                continue
            elif p == "fold_0" and self._num_folds == 1:
                #print idx, p, numpy.max(scaled_x[:, idx]), numpy.min(scaled_x[:, idx])
                assert (numpy.max(scaled_x[:, idx])-1 == 0 and
                        numpy.min(scaled_x[:, idx])-1 == 0)
            else:
                #print idx, p, numpy.max(scaled_x[:, idx]), numpy.min(scaled_x[:, idx])
                assert numpy.max(scaled_x[:, idx])-1 <= tol or \
                    abs(numpy.max(scaled_x[:, idx]) -
                        numpy.min(scaled_x[:, idx])) <= tol, \
                       (numpy.max(scaled_x[:, idx]),
                        numpy.min(scaled_x[:, idx]))
            # Check also sanity
        assert numpy.isfinite(scaled_x).all()

    def predict(self, x, tol=0.5):
        if self._model is None:
            raise ValueError("This model is not yet trained")
        # print x.shape
        x_copy = copy.deepcopy(x)
        if type(x_copy) == list or len(x_copy.shape) == 1:
            x_copy = numpy.array(x_copy, dtype="float64").reshape([1,
                                                                   len(x_copy)])
        x_copy = self._preprocess(x=x_copy)

        # Check that each input is between 0 and 1
        self._check_scaling(scaled_x=x_copy, tol=tol)
        #print self._used_param_names
        #for idx, i in enumerate(self._used_param_names):
        #    print i
        #    print x_copy[0, idx]
        return self._model.predict(x_copy)

    @property
    def scale_info(self):
        return self._scale_info