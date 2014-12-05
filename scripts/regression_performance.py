#! /bin/bash

from argparse import ArgumentParser

import os
import sys
import tempfile
import time

import numpy
RNG = 1
model_RNG = 5
numpy.random.seed(RNG)

from sklearn import cross_validation

from Surrogates.RegressionModels import ArcGP, Fastrf, GaussianProcess, GradientBoosting, KNN, LassoRegression, \
    LinearRegression, NuSupportVectorRegression, RidgeRegression, SupportVectorRegression, RandomForest, RFstruct

import Surrogates.DataExtraction.pcs_parser as pcs_parser
from Surrogates.DataExtraction.data_util import read_csv, save_one_line_to_csv, init_csv


def write_truth(train_idx, test_idx, data, fn, num_cv, dir="/tmp/"):
    train_fn = fn + "training.csv"
    test_fn = fn + "test.csv"

    _none, train_tmp = tempfile.mkstemp(suffix="training.csv_running", dir=dir)
    os.close(_none)
    _none, test_tmp = tempfile.mkstemp(suffix="test.csv_running", dir=dir)
    os.close(_none)

    fh = open(train_tmp, "w")
    fh.close()
    fh = open(test_tmp, "w")
    fh.close()
    # First train_idx
    for cv in train_idx:
        for idx, fold in enumerate(cv):
            train_truth = data[cv[idx]]
            save_one_line_to_csv(train_tmp, train_truth, len(fold))
            save_one_line_to_csv(test_tmp, data[test_idx[idx]], len(fold))
    print train_tmp, train_fn
    os.rename(train_tmp, train_fn)
    os.rename(test_tmp, test_fn)


def main():
    prog = "python whole_training.py"
    parser = ArgumentParser(description="", prog=prog)

    parser.add_argument("-d", dest="data", required=True)
    parser.add_argument("-s", dest="save", required=True)
    parser.add_argument("-r", dest="num_random", default=100, type=int,
                        help="If randomsearch is available, how many runs?")
    parser.add_argument("-m", "--model", dest="model", default="all",
                        help="Train only one model?", choices=["ArcGP", "RFstruct", "Fastrf", "GaussianProcess",
                                                               "GradientBoosting", "KNN", "LassoRegression",
                                                               "LinearRegression", "NuSupportVectorRegression",
                                                               "RidgeRegression", "SupportVectorRegression",
                                                               "RandomForest"])
    parser.add_argument("-t", "--time", dest="time", default=False,
                        action="store_true", help="Train on duration?")
    parser.add_argument("--pcs", dest="pcs", default=None, required=True,
                        help="PCS file")
    parser.add_argument("--encode", dest="encode", default=False, action="store_true")
    
    args, unknown = parser.parse_known_args()

    if args.model == "Fastrf" and args.encode:
        raise ValueError("This cannot work")

    sp = pcs_parser.read(file(args.pcs))

    model_type = args.model
    if args.encode:
        model_type += "_onehot"

    # Read data from csv
    header, data = read_csv(args.data, has_header=True, num_header_rows=3)
    para_header = header[0]
    type_header = header[1]
    cond_header = header[2]

    # Hardcoded number of crossvalidations
    num_cv = 5

    # Cut out the objective
    data_x = data[:, :-2]
    if args.time:
        print "TRAINING ON TIME"
        data_y = data[:, -2]   # -1 -> perf, -2 -> duration
    else:
        data_y = data[:, -1]   # -1 -> perf, -2 -> duration

    # Split into num_cv folds
    cv_idx = cross_validation.KFold(data_x.shape[0], n_folds=num_cv, indices=True, random_state=RNG, shuffle=True)

    # Get subsample idx
    ct = int(data_x.shape[0] / num_cv) * (num_cv-1)
    train_idx_list = list()
    test_idx_list = list()

    # For largest training set, we simply take all indices
    train_idx_list.append([train_idx for train_idx, _n in cv_idx])
    test_idx_list.extend([test_idx for _none, test_idx in cv_idx])

    # Prepare new csv
    tmp_result_header = list()
    tmp_result_header.extend([str(len(i)) for i in train_idx_list[0]])

    ct = 2000
    if ct < int(data_x.shape[0] / num_cv) * (num_cv-1):
        train_idx_list.append(list())
        for train_idx, test_idx in cv_idx:
            # NOTE: We have to change seed, otherwise trainingsamples will always be the same for different ct
            subsample_cv = cross_validation.ShuffleSplit(len(train_idx), n_iter=1, train_size=ct, test_size=None,
                                                         random_state=RNG)
            for sub_train_idx, _none in subsample_cv:
                train_idx_list[-1].append(train_idx[sub_train_idx])
                tmp_result_header.append(str(len(sub_train_idx)))

    # Now reduce in each step training set by half and subsample
    # save_ct = None
    # ct /= 2
    # if ct < 1500:
    #     save_ct = ct
    #     ct = 1500
    #
    # seed = numpy.random.randint(100, size=[num_cv])
    # while ct > 10:
    #     train_idx_list.append(list())
    #     idx = 0
    #     for train_idx, test_idx in cv_idx:
    #         # NOTE: We have to change seed, otherwise trainingsamples will always be the same for different ct
    #         subsample_cv = cross_validation.ShuffleSplit(len(train_idx), n_iter=1, train_size=ct, test_size=None,
    #                                                      random_state=seed[idx]*ct)
    #         for sub_train_idx, _none in subsample_cv:
    #             train_idx_list[-1].append(train_idx[sub_train_idx])
    #             tmp_result_header.append(str(len(sub_train_idx)))
    #         idx += 1
    #
    #     if ct > 2000 and ct/2 < 2000 and save_ct is None:
    #         # Trick to evaluate 2000 in any case
    #         save_ct = ct/2
    #         ct = 2000
    #     elif ct > 1500 and ct/2 < 1500 and save_ct is None:
    #         # Trick to evaluate 1500 in any case
    #         save_ct = ct/2
    #         ct = 1500
    #     elif save_ct is not None:
    #         ct = save_ct
    #         save_ct = None
    #     else:
    #         ct /= 2

    # Reverse train_idx to start with small dataset sizes
    train_idx_list = train_idx_list[::-1]
    result_header = ['model']
    result_header.extend(tmp_result_header[::-1])

    # print result_header
    # print [[len(j) for j in i] for i in train_idx_list]
    # print [len(i) for i in test_idx_list]

    # We could write the ground truth for this experiment
    ground_truth_fn = args.save + "ground_truth_"
    if not os.path.exists(ground_truth_fn + "training.csv") or not os.path.exists(ground_truth_fn + "test.csv"):
        write_truth(train_idx=train_idx_list, test_idx=test_idx_list, data=data_y, fn=ground_truth_fn, num_cv=num_cv, dir=args.save)

    # Now init the csv
    init_csv(args.save + '/train_duration.csv', result_header, override=False)
    init_csv(args.save + '/predict_duration.csv', result_header, override=False)

    # We need one csv containing the raw predictions
    # Just in case we already trained this model, create random filename
    if not os.path.exists(os.path.join(args.save + "prediction")):
        os.mkdir(os.path.join(args.save + "prediction"))
    _none, model_test_fn = tempfile.mkstemp(suffix=".csv_running", prefix="%s_test_prediction_" % model_type,
                                            dir=os.path.join(args.save + "prediction"))
    _none, model_train_fn = tempfile.mkstemp(suffix=".csv_running", prefix="%s_train_prediction_" % model_type,
                                             dir=os.path.join(args.save + "prediction"))

    # Now fill the array with zeros, which is fine if training failed
    train_duration_array = numpy.zeros(len(train_idx_list)*num_cv)
    predict_duration_array = numpy.zeros(len(train_idx_list)*num_cv)

    # Some variables
    train_duration = sys.maxint
    predict_duration = sys.maxint

    # Save hash to check whether we changed something during training
    data_x_hash = hash(numpy.array_repr(data_x))
    data_y_hash = hash(data_y.tostring())

    print "Train %s\n" % model_type,
    # Do all subsamples
    for train_idx_idx, train_idx_index in enumerate(train_idx_list):
        # Start training for this dataset size
        fold = -1
        for _none, test_idx in cv_idx:
            fold += 1
            current_idx = train_idx_idx*num_cv+fold
            # Start training for this fold
            sys.stdout.write("\r\t[%d | %d ]: %d" % (current_idx, len(train_idx_list)*num_cv, len(train_idx_index[fold])))
            sys.stdout.flush()
            train_data_x = numpy.array(data_x[train_idx_index[fold], :], copy=True)
            train_data_y = numpy.array(data_y[train_idx_index[fold]], copy=True)

            #num_folds = max(1, max(train_data_x[:, 0]))
            #print " Found %s folds" % num_folds

            model = fetch_model(args.model)
            model = model(rng=model_RNG, sp=sp, encode=args.encode, debug=False)

            if model.maximum_number_train_data() < train_data_x.shape[0]:
                model = None
                train_duration = numpy.nan
                predict_duration = numpy.nan
                train_predictions = numpy.zeros(train_data_x.shape[0]) * numpy.nan
                test_predictions = numpy.zeros(len(test_idx)) * numpy.nan
            else:
                train_duration = model.train(x=train_data_x, y=train_data_y, param_names=para_header[:-2])
                test_data_x = numpy.array(data_x[test_idx, :], copy=True)

                train_predictions = model.predict(x=train_data_x, tol=10)

                start = time.time()
                test_predictions = model.predict(x=test_data_x, tol=10)
                dur = time.time() - start
                predict_duration = dur
                # Also check hashes
                assert(data_y_hash == hash(data_y.tostring()) and data_x_hash == hash(numpy.array_repr(data_x)))
                del test_data_x
            del train_data_x
            del train_data_y
            del model

            train_duration_array[current_idx] = max(0, train_duration)
            predict_duration_array[current_idx] = max(0, predict_duration)

            save_one_line_to_csv(model_test_fn, test_predictions, len(train_predictions))
            save_one_line_to_csv(model_train_fn, train_predictions, len(train_predictions))

    # We're done, so remove the running from filename
    os.rename(model_train_fn, os.path.join(args.save + "prediction", "%s_train_prediction.csv" % model_type))
    os.rename(model_test_fn, os.path.join(args.save + "prediction", "%s_test_prediction.csv" % model_type))
    print "\nSaved to %s" % os.path.join(args.save + "prediction", "%s_test_prediction.csv" % model_type)

    # And save before proceeding to next model_type
    save_one_line_to_csv(args.save + '/train_duration.csv', train_duration_array, model_type)
    save_one_line_to_csv(args.save + '/predict_duration.csv', predict_duration_array, model_type)


def fetch_model(model_name):
    options = {"ArcGP": ArcGP.ArcGP,
               "RFstruct": RFstruct.RFstruct,
               "Fastrf": Fastrf.FastRF,
               "GaussianProcess": GaussianProcess.GaussianProcess,
               "GradientBoosting": GradientBoosting.GradientBoosting,
               "KNN": KNN.KNN,
               "LassoRegression": LassoRegression.LassoRegression,
               "LinearRegression": LinearRegression.LinearRegression,
               "NuSupportVectorRegression": NuSupportVectorRegression.NuSupportVectorRegression,
               "RidgeRegression": RidgeRegression.RidgeRegression,
               "SupportVectorRegression": SupportVectorRegression.SupportVectorRegression,
               "RandomForest": RandomForest.RandomForest
               }
    return options[model_name]



if __name__ == "__main__":
    main()
