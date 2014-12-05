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
import sys
from Surrogates.DataExtraction.handle_configurations import get_cat_val_map
import numpy

class RegressionBaseClass(object):

    _model = None
    _encode = None
    _name = "base class"
    _sp = None
    _scale_info = None
    _catdict = None
    _param_names = None
    _used_param_names = None
    _scale_info = None
    _rng = None
    _training_finished = False
    _num_folds = None
    _debug = False
    _max_number_train_data = sys.maxint

    def __init__(self, sp, encode, rng, debug=False):
        numpy.random.seed(rng)
        self._sp = sp
        self._rng = rng
        self._debug = debug
        self._encode = encode
        self._catdict = get_cat_val_map(self._sp)

    def train(self, **kwargs):
        raise NotImplementedError("Train is not implemented")

    def predict(self, **kwargs):
        raise NotImplementedError("Predict is not implemented")

    def maximum_number_train_data(self):
        return self._max_number_train_data

    def set_number_of_folds(self, x):
        self._num_folds = max(1, int(max(x[:, 0])+1))

    def get_sp(self):
        return copy.deepcopy(self._sp)

