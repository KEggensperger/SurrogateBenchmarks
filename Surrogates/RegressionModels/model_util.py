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

import os
import sys

import numpy as np
np.random.seed(1)

from sklearn.preprocessing import OneHotEncoder

import Surrogates.DataExtraction.configuration_space


def replace_cat_variables(x, catdict, param_names):
    #print catdict
    #print x
    new_data = np.zeros([len(x), len(x[0])]) * np.nan
    for row_idx, row in enumerate(x):
        for d_idx, d in enumerate(row):
            if param_names[d_idx] == "fold":
                new_data[row_idx][d_idx] = int(float(x[row_idx][d_idx]))
            elif param_names[d_idx] in catdict:
                if x[row_idx][d_idx] in catdict[param_names[d_idx]]:
                    # This is a regular string
                    new_data[row_idx][d_idx] = catdict[param_names[d_idx]][x[row_idx][d_idx]]
                elif str(x[row_idx][d_idx]) in catdict[param_names[d_idx]]:
                    # This is a string which got converted to a float by accident
                    new_data[row_idx][d_idx] = catdict[param_names[d_idx]][str(x[row_idx][d_idx])]
                else:
                    # This is a string which got converted to an int by accident
                    new_data[row_idx][d_idx] = catdict[param_names[d_idx]][str(int(float(x[row_idx][d_idx])))]
                    assert ((str(int(float(x[row_idx][d_idx]))) == x[row_idx][d_idx] or
                             str(float(x[row_idx][d_idx])) == x[row_idx][d_idx])
                            and type(x[row_idx][d_idx]) == str) \
                        or \
                           (int(float(x[row_idx][d_idx])) == x[row_idx][d_idx] and np.isfinite(x[row_idx][d_idx])),\
                        (x[row_idx][d_idx], param_names[d_idx])
            else:
                # here is nothing to do
                new_data[row_idx][d_idx] = float(x[row_idx][d_idx])
                assert np.isfinite(new_data[row_idx][d_idx])
    #print new_data
    assert not np.isnan(new_data).any(), new_data
    return new_data


def encode(sp, x, param_names, num_folds, catdict):
    # perform a one-hot encoding

    assert type(x) == np.ndarray

    new_param_names = list()
    new_data = None
    encode_ct = 0
    print "Encoding categorical features using a one hot encoding scheme"
    for idx_p, p in enumerate(param_names):
        if p == "duration" or p == "performance":
            continue
        elif p != "fold" and not isinstance(sp[p], Surrogates.DataExtraction.configuration_space.CategoricalHyperparameter):
            new_param_names.append(p)
            if new_data is None:
                new_data = np.array(x[:, idx_p].reshape([x.shape[0], 1]), copy=True)
            else:
                #print p, idx_p
                conc = np.array(x[:, idx_p].reshape([x.shape[0], 1]), copy=True)
                new_data = np.hstack([new_data, conc])
            continue
        #print new_data
        if p == "fold":
            encode_ct += 1
            encoder = OneHotEncoder(n_values=num_folds, categorical_features='all', dtype=int)
        elif isinstance(sp[p], Surrogates.DataExtraction.configuration_space.CategoricalHyperparameter):
            encode_ct += 1
            encoder = OneHotEncoder(n_values=len(sp[p].choices), categorical_features='all', dtype=int)
        else:
            # Do nothing
            raise ValueError("You should not be here")

        cat_values = np.array(x[:, idx_p]).reshape([x.shape[0], 1])
        #print p, cat_values
        encoder.fit(cat_values)
        encoded_values = np.array(encoder.transform(cat_values).todense())
        if new_data is None:
            new_data = encoded_values
        else:
            new_data = np.hstack([new_data, encoded_values])

        # Adjust param names
        if p == "fold":
            val_range = range(num_folds)
        else:
            val_range = range(len(sp[p].choices))

        for val in val_range:
            if p == "fold":
                new_param_names.append("%s_%d" % (p, val))
            else:
                done = False
                for value_name in sp[p].choices:
                    if catdict[p][value_name] == val:
                        new_param_names.append("%s_%s" % (p, value_name))
                        done = True
                        break
                if not done:
                    raise ValueError("Could not retranslate all categorical values")
    return new_data, new_param_names


def encode_for_ArcGP(sp, x, rel_array, param_names, num_folds, catdict):
    # perform a one-hot encoding
    assert type(x) == np.ndarray
    assert type(rel_array) == np.ndarray
    assert rel_array.shape == x.shape

    new_param_names = list()
    new_data = None
    new_rel_array = None
    encode_ct = 0
    print "Encoding categorical features using a one hot encoding scheme"
    for idx_p, p in enumerate(param_names):
        if p == "duration" or p == "performance":
            continue
        elif p != "fold" and not isinstance(sp[p], Surrogates.DataExtraction.configuration_space.CategoricalHyperparameter):
            new_param_names.append(p)
            if new_data is None:
                new_data = np.array(x[:, idx_p].reshape([x.shape[0], 1]), copy=True)
                new_rel_array = np.array(rel_array[:, idx_p].reshape([x.shape[0], 1]), copy=True)
            else:
                #print p, idx_p
                conc = np.array(x[:, idx_p].reshape([x.shape[0], 1]), copy=True)
                new_data = np.hstack([new_data, conc])
                conc = np.array(rel_array[:, idx_p].reshape([rel_array.shape[0], 1]), copy=True)
                new_rel_array = np.hstack([new_rel_array, conc])
            continue
        #print new_data
        if p == "fold":
            encode_ct += 1
            encoder = OneHotEncoder(n_values=num_folds, categorical_features='all', dtype=int)
        elif isinstance(sp[p], Surrogates.DataExtraction.configuration_space.CategoricalHyperparameter):
            encode_ct += 1
            encoder = OneHotEncoder(n_values=len(sp[p].choices), categorical_features='all', dtype=int)
        else:
            # Do nothing
            raise ValueError("You should not be here")

        cat_values = np.array(x[:, idx_p]).reshape([x.shape[0], 1])
        # print p, cat_values
        encoder.fit(cat_values)
        encoded_values = np.array(encoder.transform(cat_values).todense())
        if new_data is None:
            new_data = encoded_values
        else:
            new_data = np.hstack([new_data, encoded_values])

        # Adjust param names
        if p == "fold":
            val_range = range(num_folds) #encoder.active_features_
        else:
            val_range = range(len(sp[p].choices))

        for val in val_range:
            if new_rel_array is None:
                new_rel_array = np.array(rel_array[:, idx_p].reshape([rel_array.shape[0], 1]), copy=True)
            else:
                conc = np.array(rel_array[:, idx_p].reshape([rel_array.shape[0], 1]), copy=True)
                new_rel_array = np.hstack([new_rel_array, conc])
            if p == "fold":
                new_param_names.append("%s_%d" % (p, val))
            else:
                done = False
                for value_name in sp[p].choices:
                    if catdict[p][value_name] == val:
                        #print  (para, value_name)
                        new_param_names.append("%s_%s" % (p, value_name))
                        done = True
                        break
                if not done:
                    raise ValueError("Could not retranslate all categorical values")

    assert rel_array.shape == x.shape
    return new_data, new_rel_array, new_param_names
