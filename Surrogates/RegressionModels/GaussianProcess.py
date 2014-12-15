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


from Surrogates.RegressionModels import ScikitBaseClass

import copy
import sys
import time

import numpy
numpy.random.seed(1)

from Surrogates.DataExtraction.data_util import read_csv
from Surrogates.RegressionModels.GaussianProcess_src import gp_model, \
    hyper_parameter_sampling


class GaussianProcess(ScikitBaseClass.ScikitBaseClass):
    # Gaussian Process can inherit from scikit learn base class,
    #  as both models need the same preprocessing
    _mcmc_iters = None
    _burnin = None

    def __init__(self, sp, encode, rng, **kwargs):
        ScikitBaseClass.ScikitBaseClass.__init__(self, sp=sp, encode=encode, rng=rng, **kwargs)
        self._name = "GP " + str(encode)
        self._max_number_train_data = 2000

    def train(self, x, y, param_names, mcmc_iters=10, burnin=100,
              cov_func_name='Matern52', noiseless=False, **kwargs):
        start = time.time()
        self._mcmc_iters = mcmc_iters
        self._burnin = burnin

        scaled_x = self._set_and_preprocess(x=x, param_names=param_names)

        # Check that each input is between 0 and 1
        self._check_scaling(scaled_x=scaled_x)

        if self._debug:
            print "Shape of training data: ", scaled_x.shape
            print "Param names: ", self._used_param_names
            print "First training sample\n", scaled_x[0]
            print "Encode: ", self._encode

        # Using spearmint gp with 100xburning and 10xmcmc
        cov_func, _none = gp_model.fetchKernel(cov_func_name)

        # Initial length scales.
        ls = numpy.ones(gp_model.getNumberOfParameters(cov_func_name,
                                                       scaled_x.shape[1]))

        # Initial amplitude.
        amp2 = numpy.std(y) + 1e-4

        # Initial observation noise.
        noise = 1e-3

        #burn in
        sys.stdout.write("BURN %d times.." % burnin)
        sys.stdout.flush()
        try:
            _hyper_samples = hyper_parameter_sampling.\
                sample_hyperparameters(mcmc_iters=burnin, noiseless=noiseless,
                                       input_points=scaled_x, func_values=y,
                                       cov_func=cov_func, noise=noise,
                                       amp2=amp2, ls=ls, random_state=self._rng)

            # Now we can build the actual gp
            sys.stdout.write("MCMC %d times.." % mcmc_iters)
            sys.stdout.flush()
            (_, noise, amp2, ls) = _hyper_samples[len(_hyper_samples) - 1]

            _hyper_samples = hyper_parameter_sampling.\
                sample_hyperparameters(mcmc_iters=mcmc_iters,
                                       noiseless=noiseless,
                                       input_points=scaled_x, func_values=y,
                                       cov_func=cov_func, noise=noise,
                                       amp2=amp2, ls=ls, random_state=self._rng)

            models = list()
            sys.stdout.write("Using %d mcmc_iters" % mcmc_iters)
            sys.stdout.flush()
            for h in range(len(_hyper_samples)):
                hyper = _hyper_samples[h]
                gp = gp_model.GPModel(X=scaled_x, y=y, mean=hyper[0],
                                      noise=hyper[1], amp2=hyper[2],
                                      ls=hyper[3], cov_func=cov_func)
                models.append(gp)
                sys.stdout.write(".")
                sys.stdout.flush()
            sys.stdout.write("\n")
        except Exception, e:
            print e.message
            print "[ERROR]: Any leading minor is not positive definite, " \
                  "return None"
            models = None

        self._model = models
        duration = time.time() - start
        self._training_finished = True
        return duration

    def predict(self, x, tol=0.2, **kwargs):
        if self._model is None:
            raise ValueError("This model is not yet trained")
        # print x.shape
        x_copy = copy.deepcopy(x)
        if type(x_copy) == list or len(x_copy.shape) == 1:
            x_copy = numpy.array(x_copy, dtype="float64").reshape([1, len(x_copy)])
        x_copy = self._preprocess(x=x_copy)

        # Check that each input is between 0 and 1
        self._check_scaling(scaled_x=x_copy, tol=tol)

        assert len(self._model) == self._mcmc_iters
        print "Found %d mcmc_iters" % self._mcmc_iters
        mean_pred = list()
        for m_idx, m in enumerate(self._model):
            mean_pred.append(m.predict(x_copy, False))
        mean_pred = numpy.sum(numpy.array(mean_pred), 0)
        result = mean_pred/self._mcmc_iters
        return result


def test():
    from sklearn.metrics import mean_squared_error
    import Surrogates.DataExtraction.pcs_parser as pcs_parser
    sp = pcs_parser.read(file("/home/eggenspk/Surrogates/Data_extraction/Experiments2014/hpnnet/smac_2_06_01-dev/nips2011.pcs"))
    # Read data from csv
    header, data = read_csv("/home/eggenspk/Surrogates/Data_extraction/hpnnet_nocv_convex_all/hpnnet_nocv_convex_all_fastrf_results.csv",
                            has_header=True, num_header_rows=3)
    para_header = header[0][:-2]
    type_header = header[1]
    cond_header = header[2]
    #print data.shape
    checkpoint = hash(numpy.array_repr(data))
    assert checkpoint == 246450380584980815

    model = GaussianProcess(sp=sp, encode=False, rng=1, debug=True)
    x_train_data = data[:100, :-2]
    y_train_data = data[:100, -1]
    x_test_data = data[100:, :-2]
    y_test_data = data[100:, -1]

    model.train(x=x_train_data, y=y_train_data, param_names=para_header)

    y = model.predict(x=x_train_data[1, :])
    print "Is: %100.70f, Should: %f" % (y, y_train_data[1])
    assert y[0] == 0.470745153514900149804844886602950282394886016845703125

    print "Predict whole data"
    y_whole = model.predict(x=x_test_data)
    mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
    print "MSE: %100.70f" % mse
    assert mse == 0.006257598609004190459703664828339242376387119293212890625

    print "Soweit so gut"

    # Try the same with encoded features
    model = GaussianProcess(sp=sp, encode=True, rng=1, debug=True)
    #print data[:10, :-2]
    model.train(x=x_train_data, y=y_train_data, param_names=para_header)

    y = model.predict(x=x_train_data[1, :])
    print "Is: %100.70f, Should: %f" % (y, y_train_data[1])
    assert y[0] == 0.464671665294324409689608046392095275223255157470703125

    print "Predict whole data"
    y_whole = model.predict(x=x_test_data)
    mse = mean_squared_error(y_true=y_test_data, y_pred=y_whole)
    print "MSE: %100.70f" % mse
    assert mse == 0.00919265128042330570412588031103950925171375274658203125

    assert hash(numpy.array_repr(data)) == checkpoint

if __name__ == "__main__":
    outer_start = time.time()
    test()
    dur = time.time() - outer_start
    print "TESTING TOOK: %f sec" % dur
