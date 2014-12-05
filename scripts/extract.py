from argparse import ArgumentParser
import cPickle
import os
import csv

from sklearn.preprocessing import OneHotEncoder

import numpy as np
np.random.seed(1)

import Surrogates.DataExtraction.configuration_space
import Surrogates.DataExtraction.pcs_parser
import Surrogates.DataExtraction.handle_configurations


def build_header(sp, num_folds=1, encode=False):
    header = list()
    if encode:
        for i in range(num_folds):
            header.append("%s_%s" % ('fold', i))
    else:
        header.append("fold")

    for para in sp:
        if isinstance(sp[para],  Surrogates.DataExtraction.configuration_space.CategoricalHyperparameter):
            if encode:
                for choice in sp[para].choices:
                    header.append("%s_%s" % (para, choice))
            else:
                header.append(para)
        else:
            header.append(para)
    return header


def main():
    parser = ArgumentParser()

    parser.add_argument("-s", dest="save", default=None)
    parser.add_argument("--pcs", dest="pcs", default=None, required=True)
    args, unknown = parser.parse_known_args()

    sp = Surrogates.DataExtraction.pcs_parser.read(file(args.pcs))
    logmap = Surrogates.DataExtraction.handle_configurations.get_log_to_uniform_map(sp)
    unimap = Surrogates.DataExtraction.handle_configurations.get_uniform_to_log_map(sp)
    dflt = Surrogates.DataExtraction.handle_configurations.get_default_values(sp)
    cond_dict = Surrogates.DataExtraction.handle_configurations.get_cond_dict(sp)
    logdict = Surrogates.DataExtraction.handle_configurations.get_logparams(args.pcs)
    catdict = Surrogates.DataExtraction.handle_configurations.get_cat_val_map(sp)

    header = build_header(sp, num_folds=1, encode=False)

    # First build a csv without any encoded values
    param_data = list()
    results = list()
    durations = list()

    pkl_list = list()
    trial_ct = 0
    # To keep track of invalid results
    infinite_counter = 0
    timeout_counter = 0
    for arg in unknown:
        pkl_fn = os.path.abspath(arg)
        if not '.pkl' in pkl_fn or not os.path.exists(pkl_fn):
            raise ValueError("%s is not a .pkl file" % pkl_fn)
        print "Loading %100s" % pkl_fn,

        # Loading one pkl
        trials = cPickle.load(file(pkl_fn))

        pkl_list.append(arg)
        print "%3d" % len(trials['trials'])
        for idx, trl in enumerate(trials['trials']):
            # print idx, trl
            # Convert this trial in a regular dict
            clean_dict = dict()
            for i in trl['params'].keys():
                #print i, trl['params'][i]
                tmp_i = i                
                if i[0] == "-":
                    # Remove a trailing minus
                    tmp_i = i[1:]

                if isinstance(sp[unimap[tmp_i]], Surrogates.DataExtraction.configuration_space.CategoricalHyperparameter):
                    # This is a categorical hyperpara
                    #print tmp_i
                    clean_dict[tmp_i] = trl['params'][i].strip(" ").strip("'").strip('"') #cat[i][trl['params'][i].strip(" ").strip("'").strip('"')]
                else:
                    print tmp_i
                    clean_dict[tmp_i] = Surrogates.DataExtraction.handle_configurations.convert_to_number(trl['params'][i])
            #print "CLEAN", clean_dict
            # Unlog parameter
            clean_dict = Surrogates.DataExtraction.handle_configurations.put_on_uniform_scale(clean_dict, sp, unimap, logdict)

            #print "AFTER Unlogging:"
            #print clean_dict

            #print
            #print cond_dict
            clean_dict = Surrogates.DataExtraction.handle_configurations.remove_inactive(clean_dict, cond_dict)
            #sys.exit(1)
            #print "AFTER removing inactive:"
            #print clean_dict
            #print
            #print sp

            # Yeah we found one more config
            for fold in range(len(trl['instance_results'])):

                row = list()

                res = trl['instance_results'][fold]
                dur = trl['instance_durations'][fold]

                if res == 1 and dur < 3800:
                    timeout_counter += 1
                    continue
                #(TODO): Right now we accept also crashed runs or
                #(TODO): timeout runs which have a finite result of "max_result_on_crash"
                if not np.isfinite(res) or not np.isfinite(dur):
                    # print "%s (res) or %s (dur) is not finite" % (str(res), str(dur))
                    infinite_counter += 1
                    continue

                trial_ct += 1
                results.append(res)
                durations.append(trl['instance_durations'][fold])

                # Fold is always the first entry
                row.append(fold)
                # Now fill the param row
                for p_idx, p in enumerate(header[1:]):
                    if type(clean_dict[p]) != str and np.isnan(clean_dict[p]):
                        # Replace with default
                        if p in dflt:
                            row.append(dflt[p])
                        else:
                            raise ValueError("Don't know that param")
                    else:
                        row.append(clean_dict[p])
                param_data.append(row)
                        
    results = np.array(results)
    durations = np.array(durations)

    print "#Params: %10d" % len(header)
    print "Params:  %s" % str(header)
    print "Invalid results : %d" % infinite_counter
    print "timeout counter: %d" % timeout_counter
    print "x_data:  %10s" % str(len(param_data))
    print "results: %10d" % results.shape[0]
    print "duration:%10d" % durations.shape[0]

    # Create other headers
    type_header = list(["CAT", ])
    cond_header = list(["FALSE"])
    for p in header[1:]:
        if isinstance(sp[p],  Surrogates.DataExtraction.configuration_space.CategoricalHyperparameter):
            type_header.append("CAT")
        else:
            assert sp[p].base is None
            type_header.append("CONT")

        if p in cond_dict:
            # print ["%s=%s" % (c[0], "OR".join(str(c[1]))) for c in cond[p]]
            cond_str = "AND".join(["%s=%s" % (c[0], "OR".join([str(i) for i in c[1]])) for c in cond_dict[p]])
            cond_header.append(cond_str)
        else:
            cond_header.append("FALSE")

    # Put together all data
    for row_idx, row in enumerate(param_data):
        param_data[row_idx].append(durations[row_idx])
        param_data[row_idx].append(results[row_idx])


    header.append('duration')
    header.append('performance')
    type_header.append('TARGET')
    type_header.append('TARGET')
    cond_header.append('TARGET')
    cond_header.append('TARGET')


    print "-------"
    print
    print
    print "%s" % (",".join(["%12s" % i[:12] for i in header]))
    print "%s" % (",".join(["%12s" % i[:12] for i in type_header]))
    print "%s" % (",".join(["%12s" % i for i in cond_header]))
    for row in range(10):
        print "%s" % (",".join(["%12.5s" % str(i) for i in param_data[row]]))
    print "..."

    if args.save is not None:
        fn = args.save + '_fastrf_results.csv'
        print "Save to %s" % fn
        with open(fn, 'wb') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow([i for i in header])
            result_writer.writerow([i for i in type_header])
            result_writer.writerow([i for i in cond_header])

            for row in range(results.shape[0]):
                line = ["%s" % str(i) for i in param_data[row]]
                result_writer.writerow(line)


    # Now replace cat params with numbers
    for row_idx, row in enumerate(param_data):
        for p_idx, p in enumerate(header):
            if type_header[p_idx] == "TARGET" or p == "fold" or type_header[p_idx] == "CONT":
                continue
            elif isinstance(sp[p],  Surrogates.DataExtraction.configuration_space.CategoricalHyperparameter):
                param_data[row_idx][p_idx] = catdict[p][param_data[row_idx][p_idx]]
            else:
                raise ValueError("Don't know that para type: %s" % type_header[p_idx])
    param_data = np.array(param_data)

    # Now write down spear_cond version
    print "-------"
    print
    print
    print "%s" % (",".join(["%12s" % i[:12] for i in header]))
    print "%s" % (",".join(["%12s" % i[:12] for i in type_header]))
    print "%s" % (",".join(["%12s" % i for i in cond_header]))
    for row in range(10):
        print "%s" % (",".join(["%12.5s" % str(i) for i in param_data[row]]))
    print "..."

    if args.save is not None:
        fn = args.save + '_spear_cond_results.csv'
        print "Save to %s" % fn
        with open(fn, 'wb') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow([i for i in header])
            result_writer.writerow([i for i in type_header])
            result_writer.writerow([i for i in cond_header])

            for row in range(results.shape[0]):
                line = ["%s" % str(i) for i in param_data[row]]
                result_writer.writerow(line)


    checkpoint = np.array(param_data[:, -2:], copy=True)

    print "Encoding categorical features using a one hot encoding scheme"
    new_data = None
    new_para_header = list()
    new_type_header = list()
    new_cond_header = list()
    encode_ct = 0
    for p_idx, para in enumerate(header):
        if cond_header[p_idx] == "TARGET":
            new_para_header.append(para)
            new_type_header.append("TARGET")
            new_cond_header.append("TARGET")
            conc = np.array(param_data[:, p_idx].reshape([param_data.shape[0], 1]), copy=True)
            new_data = np.hstack([new_data, conc])
            continue

        # First define new value for cond_header
        if cond_header[p_idx] == "FALSE":
            new_cond = "FALSE"
        else:
            tmp_cond = cond_header[p_idx].split("AND")
            new_cond = list()
            for c in tmp_cond:
                c = c.split("=")
                c_cond = c[0]
                pos_val = c[1].split("OR")
                cond_names = "OR".join(["%s_%s" % (c_cond, i) for i in pos_val])
                new_cond.append("%s=1" % str(cond_names))
            new_cond = "AND".join(new_cond)

        if type_header[p_idx] != "CAT":
            new_para_header.append(para)
            new_type_header.append("CONT")
            new_cond_header.append(new_cond)
            if new_data is None:
                new_data = np.array(param_data[:, p_idx], copy=True)
            else:
                conc = np.array(param_data[:, p_idx].reshape([param_data.shape[0], 1]), copy=True)
                new_data = np.hstack([new_data, conc])
            continue

        encode_ct += 1
        # print "%s is a CAT para" % para
        if para == "fold":
            encoder = OneHotEncoder(n_values='auto', categorical_features='all', dtype=int)
        else:
            encoder = OneHotEncoder(n_values=len(catdict[para]), categorical_features='all', dtype=int)  

        cat_values = np.array(param_data[:, p_idx]).reshape([param_data.shape[0], 1])
        encoder.fit(cat_values)
        encoded_values = np.array(encoder.transform(cat_values).todense())
        if new_data is None:
            new_data = encoded_values
        else:
            new_data = np.hstack([new_data, encoded_values])
        # print encoder.active_features_

        if para == "fold":
            val_range = encoder.active_features_
        else:
            val_range = range(len(catdict[para].keys()))

        for val in val_range:
            if para == "fold":
                new_para_header.append("%s_%d" % (para, val))
            else:
                done = False
                for value_name in catdict[para]:
                    if catdict[para][value_name] == val:
                        #print  (para, value_name)
                        new_para_header.append("%s_%s" % (para, value_name))
                        done = True
                        break
                if not done:
                    raise ValueError("Could not retranslate all categorical values")
            
        #new_para_header.extend(["%s_%d" % (para, val) for val in encoder.active_features_])
        new_type_header.extend(["CAT" for val in val_range]) #encoder.active_features_])
        new_cond_header.extend([new_cond for val in val_range]) #encoder.active_features_])

    num_vals = sum([1 if i == "CAT" else 0 for i in new_type_header])
    print "Encoding %d of %d features with %d values" % (encode_ct, len(header), num_vals)
    assert new_data.shape[1] == len(header)-encode_ct + num_vals,\
        "Data shape is %f, but it should be %f" % (new_data.shape[1], len(header)-encode_ct + num_vals)

    # Overwrite old data
    header = new_para_header
    type_header = new_type_header
    cond_header = new_cond_header
    param_data = new_data

    # Check whether we still have the same data
    assert ((checkpoint == new_data[:, -2:]).all())

    # Now write down encoded version
    print "-------"
    print
    print
    print "%s" % (",".join(["%12s" % i[:12] for i in header]))
    print "%s" % (",".join(["%12s" % i[:12] for i in type_header]))
    print "%s" % (",".join(["%12s" % i for i in cond_header]))
    for row in range(10):
        print "%s" % (",".join(["%12.5s" % str(i) for i in param_data[row]]))
    print "..."

    if args.save is not None:
        fn = args.save + '_encoded_results.csv'
        print "Save to %s" % fn
        with open(fn, 'wb') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow([i for i in header])
            result_writer.writerow([i for i in type_header])
            result_writer.writerow([i for i in cond_header])

            for row in range(results.shape[0]):
                line = ["%s" % str(i) for i in param_data[row]]
                result_writer.writerow(line)


if __name__ == "__main__":
    main()

