from argparse import ArgumentParser
import cPickle
import os

import numpy
from sklearn import cross_validation

RNG = 1
numpy.random.seed(RNG)

from Surrogates.DataExtraction.data_util import read_csv
from Surrogates.DataExtraction import pcs_parser
# from Surrogates.RegressionModels import ArcGP, RFstruct
from Surrogates.RegressionModels import GradientBoosting, KNN, LassoRegression
from Surrogates.RegressionModels import LinearRegression, RidgeRegression
from Surrogates.RegressionModels import SupportVectorRegression #, GaussianProcess,
from Surrogates.RegressionModels import RandomForest, NuSupportVectorRegression

__author__ = 'eggenspk'


def main():
    prog = "python train.py"
    parser = ArgumentParser(description="", prog=prog)

    # Data stuff for training surrogate
    parser.add_argument("-m", "--model", dest="model", default=None, required=True,
                        help="What model?",
                        choices=[#"ArcGP", "RFstruct",
                                 # "GaussianProcess",
                                 "GradientBoosting", "KNN", "LassoRegression",
                                 "LinearRegression", "SupportVectorRegression",
                                 "RidgeRegression", "NuSupportVectorRegression",
                                 "RandomForest"])
    parser.add_argument("--data", dest="data_fn", default=None, required=True,
                        help="Where is the csv with training data?")
    parser.add_argument("--pcs", dest="pcs", default=None, required=False,
                        help="Smac pcs file for this experiment")
    parser.add_argument("--encode", dest="encode", default=False,
                        action="store_true")
    parser.add_argument("--saveto", dest="saveto", required=True)

    args, unknown = parser.parse_known_args()

    if os.path.exists(args.saveto):
        raise ValueError("%s already exists" % args.saveto)
    if not os.path.isdir(os.path.dirname(args.saveto)):
        raise ValueError("%s, directory does not exist")

    saveto = os.path.abspath(args.saveto)

    if args.model == "Fastrf" and args.encode:
        raise ValueError("This cannot work")

    sp = pcs_parser.read(file(args.pcs))

    model_type = args.model
    if args.encode:
        model_type += "_onehot"

    # Read data from csv
    header, data = read_csv(args.data_fn, has_header=True, num_header_rows=3)
    para_header = header[0][:-2]
    #type_header = header[1][:-2]
    #cond_header = header[2][:-2]

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
        train_data_x, _n_x, train_data_y, _n_y = \
            cross_validation.train_test_split(train_data_x, train_data_y,
                                              train_size=max_n,
                                              random_state=RNG)
        print "to %d" % train_data_x.shape[0]
    else:
        print "Reducing data not neccessary"

    dur = model.train(x=train_data_x, y=train_data_y, param_names=para_header)

    print "Training took %fsec" % dur

    if args.model == "Fastrf" or "RFstruct":
        # We need to save the forest
        print "Saved forest to %s" % saveto
        model.save_forest(fn=saveto + "_forest")

    assert data_x_hash, hash(numpy.array_repr(data_x))
    assert data_y_hash, hash(data_y.tostring())

    fn = open(saveto, "wb")
    cPickle.dump(obj=model, file=fn, protocol=cPickle.HIGHEST_PROTOCOL)
    fn.close()
    print "Saved to %s" % saveto


def fetch_model(model_name):
    options = {#"ArcGP": ArcGP.ArcGP,
               #"RFstruct": RFstruct.RFstruct,
               #"GaussianProcess": GaussianProcess.GaussianProcess,
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