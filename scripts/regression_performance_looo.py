#! /bin/bash

from argparse import ArgumentParser

import os
import tempfile

import numpy
RNG = 1
model_RNG = 1
numpy.random.seed(RNG)

from sklearn import cross_validation

from Surrogates.RegressionModels import ArcGP, Fastrf, GaussianProcess
from Surrogates.RegressionModels import GradientBoosting, KNN, LassoRegression
from Surrogates.RegressionModels import LinearRegression, RidgeRegression
from Surrogates.RegressionModels import NuSupportVectorRegression, RandomForest
from Surrogates.RegressionModels import SupportVectorRegression, RFstruct

import Surrogates.DataExtraction.pcs_parser as pcs_parser
from Surrogates.DataExtraction.data_util import read_csv, save_one_line_to_csv

"""
# Comment this out, till we can be sure it is not needed
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
"""

def main():
    prog = "python whole_training.py"
    parser = ArgumentParser(description="", prog=prog)

    parser.add_argument("--traindata", dest="traindata", required=True)
    parser.add_argument("--testdata", dest="testdata", required=True)
    parser.add_argument("-s", dest="save", required=True)
    parser.add_argument("-r", dest="num_random", default=100, type=int,
                        help="If randomsearch is available, how many runs?")
    parser.add_argument("-m", "--model", dest="model", default="all",
                        help="Train only one model?", choices=["ArcGP", "RFstruct", "Fastrf", "GaussianProcess",
                                                               "GradientBoosting", "KNN", "LassoRegression",
                                                               "LinearRegression", "NuSupportVectorRegression",
                                                               "RidgeRegression", "SupportVectorRegression",
                                                               "RandomForest"])
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
    header, data = read_csv(args.traindata, has_header=True, num_header_rows=3)
    para_header = header[0]
    type_header = header[1]
    cond_header = header[2]

    # Cut out the objective
    data_x = data[:, :-2]
    data_y = data[:, -1]   # -1 -> perf, -2 -> duration

    # Save hash to check whether we changed something during training
    data_x_hash = hash(numpy.array_repr(data_x))
    data_y_hash = hash(data_y.tostring())

    print "Train %s\n" % model_type,
    train_data_x = numpy.array(data_x, copy=True)
    train_data_y = numpy.array(data_y, copy=True)

    model = fetch_model(args.model)
    model = model(rng=RNG, sp=sp, encode=args.encode, debug=False)

    if model.maximum_number_train_data() < train_data_x.shape[0]:
        max_n = model.maximum_number_train_data()
        print "Limited model, reducing #data from %d" % train_data_x.shape[0]
        train_data_x, _n_x, train_data_y, _n_y = cross_validation.train_test_split(train_data_x, train_data_y,
                                                                                   train_size=max_n,
                                                                                   random_state=RNG)
        print "to %d" % train_data_x.shape[0]
    else:
        print "Reducing data not neccessary"

    dur = model.train(x=train_data_x, y=train_data_y, param_names=para_header[:-2])

    print "Training took %fsec" % dur

    _header, test_data = read_csv(args.testdata, has_header=True, num_header_rows=3)
    assert para_header == _header[0]
    assert type_header == _header[1]
    assert cond_header == _header[2]
     # Cut out the objective
    test_data_x = test_data[:, :-2]
    test_predictions = model.predict(x=test_data_x, tol=10)

    model_test_fn = os.path.join(args.save, "test_prediction.csv")
    # Dirty hack to initialize, because it's quite late
    if not os.path.isfile(model_test_fn):
        fh = open(model_test_fn, "w")
        fh.close()

    print test_predictions.shape
    save_one_line_to_csv(model_test_fn, test_predictions, model_type)


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