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


import sys

import numpy
numpy.random.seed(1)

import Surrogates.DataExtraction.configuration_space


def scale(scale_info, x):
    #print scale_info
    lower, upper = scale_info
    if lower is None and upper is None:
        raise ValueError("Here is nothing to scale")

    #print scale_info
    #print len(x[0]), len(lower)
    assert len(x[0]) == len(lower), (len(x[0]), len(lower))

    for row_idx, row in enumerate(x):
        for p_idx, p in enumerate(row):
            if lower[p_idx] is None:
                assert upper[p_idx] is None
                continue
            try:
                x[row_idx][p_idx] -= lower[p_idx]
                x[row_idx][p_idx] /= (upper[p_idx] - lower[p_idx])
            except TypeError:
                sys.stderr.write("Error: %s, %s, %s" %(x[row_idx][p_idx],
                                                       lower[p_idx],
                                                       upper[p_idx]))
                raise
    return x


def get_x_info_scaling_all(sp, x, param_names, num_folds, encoded=False):
    # We scale x values to be in the range [0, 1]
    min_val = list()
    max_val = list()
    #print param_names, sp
    for idx_p, p in enumerate(param_names):
        #print p
        if p == "performance" or p == "duration":
            continue

        if p not in sp and encoded:
            # Encoded para
            enc_p = "_".join(p.split("_")[:-1])
            assert enc_p == 'fold' or enc_p in sp, p
            min_val.append(None)
            max_val.append(None)
            continue
        else:
            assert p == 'fold' or p in sp, p

        if p == "result" or p == "duration":
            break

        if p == 'fold':
            if num_folds == 1:
                # If we have one fold, we do nothing
                min_val.append(None)
                max_val.append(None)
            else:
                min_val.append(0)
                max_val.append(num_folds-1)
        elif isinstance(sp[p], Surrogates.DataExtraction.configuration_space.
                        CategoricalHyperparameter):
            # We scale features to be within [0,1]
            min_val.append(0)
            len_choices = len(sp[p].choices)-1
            if len_choices == 0:
                len_choices = 1
            max_val.append(len_choices)
        else:
            # We have a continuous para
            # Just to double check
            assert sp[p].base is None
            tmp_min = float(numpy.min(x[:, idx_p], axis=0))
            tmp_max = float(numpy.max(x[:, idx_p], axis=0))
            assert tmp_max > tmp_min
            assert numpy.isfinite(tmp_min)
            assert numpy.isfinite(tmp_max)
            min_val.append(tmp_min)
            max_val.append(tmp_max)
    return min_val, max_val


def get_x_info_scaling_no_categorical(sp, x, param_names, num_folds):
    # We scale x values to be in the range [0, 1]
    min_val = list()
    max_val = list()
    #print param_names, sp
    for idx_p, p in enumerate(param_names):
        #print p
        if p == "performance" or p == "duration":
            continue

        if p == 'fold':
                min_val.append(None)
                max_val.append(None)
        elif isinstance(sp[p], Surrogates.DataExtraction.configuration_space.
                        CategoricalHyperparameter):
            # We scale features to be within [0,1]
            min_val.append(None)
            max_val.append(None)
        else:
            # We have a continuous para
            # Just to double check
            assert sp[p].base is None
            tmp_min = float(numpy.min(x[:, idx_p], axis=0))
            tmp_max = float(numpy.max(x[:, idx_p], axis=0))
            assert tmp_max > tmp_min
            assert numpy.isfinite(tmp_min)
            assert numpy.isfinite(tmp_max)
            min_val.append(tmp_min)
            max_val.append(tmp_max)
    return min_val, max_val