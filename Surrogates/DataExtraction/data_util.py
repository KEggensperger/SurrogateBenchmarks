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

import csv
import os
import sys


def read_csv(fn, has_header=True, num_header_rows=1):
    import numpy as np

    header_rows_left = num_header_rows
    if not has_header:
        header_rows_left = 0

    data = list()
    header = list()
    with open(fn, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csv_reader:
            if header_rows_left > 0:
                header_rows_left -= 1
                header.append(row)
                continue
            data.append([float(i.strip()) for i in row])
    data = np.array(data)

    if not has_header:
        header = None
    if num_header_rows == 1:
        header = header[0]

    return header, data

def read_csv_as_list(fn, has_header=True, num_header_rows=1):
    import numpy as np

    header_rows_left = num_header_rows
    if not has_header:
        header_rows_left = 0

    data = list()
    header = list()
    with open(fn, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csv_reader:
            if header_rows_left > 0:
                header_rows_left -= 1
                header.append(row)
                continue
            row_list = list()
            for v_idx, val in enumerate(row):
                try:
                    val = float(val.strip())
                    if int(val) == val:
                        val = int(val)
                except:
                    val = val.strip()
                row_list.append(val)
            data.append(row_list)
    #data = np.array(data)

    if not has_header:
        header = None
    if num_header_rows == 1:
        header = header[0]

    return header, data


def write_to_csv(fn, header=None, data=None):
    with open(fn, 'wb') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if header is not None:
            csv_writer.writerow(header)
        if data is not None:
            for row in range(len(data)):
                csv_writer.writerow(data[row])


# ============== MISC

def init_csv(fn, header, override=False):
    if os.path.isfile(fn) and not override:
        return
    with open(fn, 'wb') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(header)


# =============== READ

def read_data(fn, has_header=True):
    import numpy
    data = list()
    header = None
    with open(fn, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for idx, row in enumerate(csv_reader):
            if header is None and has_header:
                header = row
                continue
            try:
                data.append([numpy.nan if i == "nan" else float(i.strip()) for i in row])
            except ValueError:
                print idx, row
                sys.exit(1)
    return header, data


def read_key_value(fn, has_header=True):
    import numpy
    data = dict()
    header = None
    with open(fn, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for idx, row in enumerate(csv_reader):
            if header is None and has_header:
                header = row
                continue
            try:
                data[row[0]] = [numpy.nan if i == "nan" else float(i.strip()) for i in row[1:]]
            except ValueError:
                print idx, row
                sys.exit(1)
    return header, data


def read_matrix_replace_higher_values(fn,  max_value=sys.maxint, has_header=True):
    import numpy as np
    data = dict()
    key_list = list()
    header = None
    with open(fn, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csv_reader:
            if header is None and has_header:
                header = row
                continue
            key_list.append(row[0].strip())
            data[row[0].strip()] = [min(max_value, float(i.strip())) if not np.isnan(float(i.strip()))
                                    else np.NaN for i in row[1:]]
    return header, data, key_list


# =================== WRITE

def save_one_line_to_csv(fn, data, model):
    with open(fn, 'ab') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        tmp = [model, ]
        tmp.extend([str(value) for value in data])
        csv_writer.writerow(tmp)


def save_key_data_to_csv(fn, data, keys=None, header=None):
    with open(fn, 'wb') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if header is not None:
            csv_writer.writerow(header)
        for idx in range(data.shape[0]):
            tmp = list()
            if keys is not None:
                tmp.append(keys[idx])
            tmp.extend(data[idx, :].flatten())
            csv_writer.writerow(tmp)


def save_to_csv(fn, header, data, models, key):
    with open(fn, 'wb') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(header)
        for idx, row in enumerate(models):
            tmp = [row, ]
            tmp.extend([str(value) for value in data[row][key]])
            csv_writer.writerow(tmp)


def write_matrix_to_csv(fn, matrix, header=None):
    with open(fn, 'wb') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if header is not None:
            csv_writer.writerow(header)
        for row in range(len(matrix)):
            csv_writer.writerow(matrix[row])

            
# ==================READ FROM PKL

def get_Trace_cv(trials):
    import numpy as np
    trace = list()
    trials_list = trials['trials']
    instance_order = trials['instance_order']
    instance_mean = np.ones([len(trials_list), 1]) * np.inf
    instance_val = np.ones([len(trials_list), len(trials_list[0]['instance_results'])]) * np.nan
    for tr_idx, in_idx in instance_order:
        instance_val[tr_idx, in_idx] = trials_list[tr_idx]['instance_results'][in_idx]
        
        val = nan_mean(instance_val[tr_idx, :])
        if np.isnan(val):
            val = np.inf
        instance_mean[tr_idx] = val
        trace.append(np.min(instance_mean, axis=0)[0])
    if np.isnan(trace[-1]):
        del trace[-1]
    return trace
    

def nan_mean(arr):
    import numpy as np
    # First: Sum all finite elements
    arr = np.array(arr)
    res = sum([ele for ele in arr if np.isfinite(ele)])
    num_ele = (arr.size - np.count_nonzero(~np.isfinite(arr)))
    if num_ele == 0:
        return np.nan
    if num_ele != 0 and res == 0:
        return 0
    # Second: divide with number of finite elements
    res /= num_ele
    return res
