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

import numpy as np
import re
import configuration_space
from collections import deque, OrderedDict


def remove_param_metadata(params):
    """
    Check whether some params are defined on the Log scale or with a Q value,
    must be marked with "LOG$_{paramname}" or Q[0-999]_$paramname
    LOG/Q will be removed from the paramname
    """
    for para in params.keys():
        tmp = params[para]
        del params[para]
        #para = para.strip().replace("-", "")
        para = para.strip()
        if para[0] == "-":
            para = para[1:]
        params[para] = tmp
        if "LOG10_" in para:
            pos = para.find("LOG10")
            new_name = para[0:pos] + para[pos + 6:]
            # new_name = new_name.strip("_")
            params[new_name] = np.power(10, float(params[para]))
            del params[para]
        elif "LOG2" in para:
            pos = para.find("LOG2_")
            new_name = para[0:pos] + para[pos + 5:]
            # new_name = new_name.strip("_")
            params[new_name] = np.power(2, float(params[para]))
            del params[para]
        elif "LOG_" in para:
            pos = para.find("LOG")
            new_name = para[0:pos] + para[pos + 4:]
            # new_name = new_name.strip("_")
            params[new_name] = np.exp(float(params[para]))
            del params[para]

            #Check for Q value, returns round(x/q)*q
        m = re.search(r'Q[0-999\.]{1,10}_', para)
        if m is not None:
            pos = new_name.find(m.group(0))
            tmp = new_name[0:pos] + new_name[pos + len(m.group(0)):]
            #tmp = tmp.strip("_")
            q = float(m.group(0)[1:-1])
            params[tmp] = round(float(params[new_name]) / q) * q
            del params[new_name]

    for para in params.keys():
        assert (type(params[para]) == str or np.isfinite(params[para]))

    return params


def convert_to_number(strng):
    if type(strng) == str:
        strng = strng.strip().strip("'")
    a = float(strng)
    try:
        b = int(strng)
    except ValueError:
        b = np.nan
        pass
    if a == b and np.isfinite(b):
        return b
    elif np.isfinite(a):
        return a
    else:
        raise ValueError("%s is not a number" % strng)


def get_logparams(pcs_fn):
    fh = open(pcs_fn, 'r')
    log_dict = dict()

    for line in fh.readlines():
        if '#' in line:
            line = "".join(line.split('#')[:-1])
        line = line.strip("\n").strip(" ")
        if len(line) == 0:
            continue

        if "|" in line:
            continue
        #print line
        if "{" in line:
            if "LOG" in line:
                raise ValueError("A categorical para on a logscale?: %s" % line)
        elif "ondi" not in line:
            complete_line = line
            para_name = line.split("[")[0].strip()

            if "LOG10_" in para_name:
                pos = para_name.find("LOG10_")
                new_name = remove_param_metadata({para_name: 23}).keys()[0]
                log_dict[new_name] = "LOG10"
            elif "LOG2_" in para_name:
                pos = para_name.find("LOG2_")
                new_name = remove_param_metadata({para_name: 23}).keys()[0]
                log_dict[new_name] = "LOG2"
            elif "LOG_" in para_name:
                pos = para_name.find("LOG_")
                new_name = remove_param_metadata({para_name: 23}).keys()[0]
                log_dict[new_name] = "LOG"
            elif "]l" in complete_line or "]il" in complete_line or "]li" in complete_line:
                #log_dict[para_name] = "POW10"
                raise ValueError("Please fix me: %s " % line)
            else:
                # This is a regular para
                continue
        else:
            print "Don't know what is this: %s" % line
            raise ValueError(line)
    return log_dict


def remove_inactive(clean_dict, cond):
    replaced_pars = list()
    q = deque(cond.keys())

    while len(q) > 0:
        param_that_might_be_inactive = q.popleft()
        # First thing to check: Is this para already in dict or do we have to replace it?
        if param_that_might_be_inactive not in clean_dict:
            # Then it will never become active
            # print "KhkhkH", param_that_might_be_inactive
            pass

        # Check each condition
        act = True
        for dep in cond[param_that_might_be_inactive]:
            cond_para, cond_value = dep
            #print dep
            # print dep, param_that_might_be_inactive
            # First check whether we depend on a dependent para that has been replaced
            # Then out para is also not active
            if cond_para in cond and cond_para in replaced_pars and clean_dict[cond_para] == np.nan:
                act = False
                break
            elif cond_para in cond and cond_para not in replaced_pars:
                # if yes put it back to queue, we have to wait
                act = None
                q.append(param_that_might_be_inactive)
                break

            # Then it must be already in the dict
            if cond_para in clean_dict:
                # Two things can happen
                if type(clean_dict[cond_para]) != str and np.isnan(clean_dict[cond_para]):
                    # 1. para depends on a non active para, then it is automatically inactive
                    act = False
                    break
                else:
                    #print act
                    # 2. it is a regular dependency, then we have to check it
                    #print type(clean_dict[cond_para]), [val for val in cond_value]
                    if type(cond_value) != list:
                        if clean_dict[cond_para] == cond_value:
                            is_condition_true = 1
                        else:
                            is_condition_true = 0
                    else:
                        is_condition_true = sum([1 if clean_dict[cond_para] == val else 0 for val in cond_value])
                    #print is_condition_true, type(clean_dict[cond_para]), type(cond_value[0])
                    if is_condition_true > 1:
                        raise ValueError("More than one condition "
                                         "is true: %s, %s" % (str(clean_dict), str(cond[param_that_might_be_inactive])))
                    elif is_condition_true == 1:
                        act = (act and True)
                    elif is_condition_true == 0:
                        act = False
                        break
                    else:
                        raise ValueError("FIXME")

            else:
                raise ValueError("Could not find %s" % cond_para)

        if act is not None:
            if not act:
                clean_dict[param_that_might_be_inactive] = np.nan
            else:
                #print clean_dict
                #print param_that_might_be_inactive
                clean_dict[param_that_might_be_inactive] = clean_dict[param_that_might_be_inactive]
            replaced_pars.append(param_that_might_be_inactive)
    return clean_dict


def get_log_to_uniform_map(sp):
    logmap = dict()
    for para in sp:
        logmap[para] = remove_param_metadata({para: 23}).keys()[0]
    return logmap


def get_uniform_to_log_map(sp):
    unimap = dict()
    for para in sp:
        unimap[remove_param_metadata({para: 23}).keys()[0]] = para
    return unimap


def get_default_values(sp):
    dflt = dict()
    for para in sp:
        if len(sp[para].conditions[0]) == 0:
            # print "Has no conditions"
            dflt[para] = np.nan
        elif isinstance(sp[para], configuration_space.CategoricalHyperparameter):
            dflt[para] = sp[para].choices[0]
        else:
            dflt[para] = (sp[para].upper + sp[para].lower)/2.0
    return dflt


def get_cat_val_map(sp):
    cat_dict = OrderedDict()
    for para in sp:
        if isinstance(sp[para], configuration_space.CategoricalHyperparameter):
            cat_dict[para] = OrderedDict()
            choices = list(sp[para].choices)
            choices.sort()
            for i in range(len(choices)):
                cat_dict[para][choices[i]] = i
    return cat_dict


def put_on_uniform_scale(clean_dict, sp, unimap, logdict):
    new_dict = dict()
    for key in clean_dict:
        if isinstance(sp[unimap[key]], configuration_space.CategoricalHyperparameter):
            # Do nothing as this is a categorical para
            new_dict[key] = clean_dict[key]
        elif key not in logdict:
            # Again do nothing as this is a uniform base
            new_dict[key] = clean_dict[key]
        elif logdict[key] == "LOG10":
            new_dict[unimap[key]] = np.log10(clean_dict[key])
        elif logdict[key] == "LOG2":
            new_dict[unimap[key]] = np.log2(clean_dict[key])
        elif logdict[key] == "LOG":
            new_dict[unimap[key]] = np.log(clean_dict[key])
        else:
            raise ValueError("This param is on a not known base: %s, %s" % (key, logdict[key]))
    return new_dict


def get_cond_dict(sp):
    cond_dict = dict()
    for para in sp:
        if len(sp[para].conditions[0]) == 0:
            # Nothing to do here
            pass
        elif len(sp[para].conditions) > 1:
            raise ValueError("Cannot handle multiple or conditions")
        else:
            cond_dict[para] = list()
            dep = para
            for and_cond in sp[para].conditions[0]:
                parent, eq, val = and_cond.split(" ")
                assert parent in sp
                assert isinstance(sp[parent], configuration_space.CategoricalHyperparameter)
                if eq == "==":
                    cond_dict[para].append([parent, val])
                elif eq == "in":
                    val = val.strip("}").strip("{")
                    cond_dict[para].append([parent, val.split(",")])
                else:
                    raise ValueError("Don't know that: %s" % eq)
    return cond_dict