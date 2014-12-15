'''
Created on 12.12.2013

@author: Simon Bartels

This class wraps the hyper-parameter sampling methods for the Gaussian processes of spear mint.
'''

import numpy as np
np.random.seed(1)

import scipy.linalg as spla
from spearmint.util import slice_sample
from spearmint.helpers import log

'''
Global constants.
'''
NOISE_SCALE = 0.1  # horseshoe prior
AMP2_SCALE  = 1    # zero-mean log normal prior
MAX_LS      = 2    # top-hat prior on length scales


def sample_from_proposal_measure(starting_point, log_proposal_measure, number_of_points):
    '''
    Samples representer points for discretization of Pmin.
    Args:
        starting_point: The point where to start the sampling.
        log_proposal_measure: A function that measures in log-scale how suitable a point is to represent Pmin.
        number_of_points: The number of samples to draw.
    Returns:
        a numpy array containing the desired number of samples
    '''
    representer_points = np.zeros([number_of_points, starting_point.shape[0]])
    chain_length = 3 * starting_point.shape[0]
    #TODO: burnin?
    for i in range(0, number_of_points):
        #this for loop ensures better mixing
        for c in range(0, chain_length):
            try:
                starting_point = slice_sample(starting_point, log_proposal_measure)
            except Exception as e:
                starting_point = handle_slice_sampler_exception(e, starting_point, log_proposal_measure)
        representer_points[i] = starting_point
    return representer_points

'''
How often we try to restart the slice sampler if it shrank to zero.
'''
NUMBER_OF_RESTARTS = 3


def handle_slice_sampler_exception(exception, starting_point, proposal_measure, opt_compwise=False):
    '''
    Handles slice sampler exceptions. If the slice sampler shrank to zero the slice sampler will be restarted
    a few times. If this fails or if the exception was another this method will raise the given exception.
    Args:
        exception: the exception that occured
        starting_point: the starting point that was used
        proposal_measure: the used proposal measure
        opt_compwise: how to set the compwise option
    Returns:
        the output of the slice sampler
    Raises:
        Exception: the first argument
    '''
    if exception.message == "Slice sampler shrank to zero!":
        log("Slice sampler shrank to zero! Action: trying to restart " + str(NUMBER_OF_RESTARTS)
            + " times with same starting point")
        restarts_left = NUMBER_OF_RESTARTS
        while restarts_left > 0:
            try:
                return slice_sample(starting_point, proposal_measure, compwise=opt_compwise)
            except Exception as e:
                log("Restart failed. " + str(restarts_left) + " restarts left. Exception was: " + e.message)
                restarts_left = restarts_left - 1
        # if we leave the while loop we will raise the exception we got
    import traceback
    print traceback.format_exc()
    raise exception


def _sample_mean_amp_noise(comp, vals, cov_func, start_point, ls):
    default_noise = 1e-3
    #if we get a start point that consists only of two variables that means we don't care for the noise
    noiseless = (start_point.shape[0] == 2)
    def logprob(hypers):
        mean = hypers[0]
        amp2 = hypers[1]
        if not noiseless:
            noise = hypers[2]
        else:
            noise = default_noise

        # This is pretty hacky, but keeps things sane.
        if mean > np.max(vals) or mean < np.min(vals):
            return -np.inf

        if amp2 < 0 or noise < 0:
            return -np.inf
        
        cov = (amp2 * (cov_func(ls, comp, None) + 
            1e-6 * np.eye(comp.shape[0])) + noise * np.eye(comp.shape[0]))
        chol = spla.cholesky(cov, lower=True)
        solve = spla.cho_solve((chol, True), vals - mean)
        lp = -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(vals - mean, solve)
        if not noiseless:
            # Roll in noise horseshoe prior.
            lp += np.log(np.log(1 + (NOISE_SCALE / noise) ** 2))

        # Roll in amplitude lognormal prior
        lp -= 0.5 * (np.log(amp2) / AMP2_SCALE) ** 2
        #print "LP: " + str(lp)
        return lp
    try:
        return slice_sample(start_point, logprob, compwise=False)
    except Exception as e:
        return handle_slice_sampler_exception(e, start_point, logprob, False)

def _sample_ls(comp, vals, cov_func, start_point, mean, amp2, noise):
    def logprob(ls):
        if np.any(ls < 0) or np.any(ls > MAX_LS):
            return -np.inf

        cov = (amp2 * (cov_func(ls, comp, None) + 
            1e-6 * np.eye(comp.shape[0])) + noise * np.eye(comp.shape[0]))
        chol = spla.cholesky(cov, lower=True)
        solve = spla.cho_solve((chol, True), vals - mean)

        lp = (-np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(vals - mean, solve))
        return lp

    try:
        return slice_sample(start_point, logprob, compwise=True)
    except Exception as e:
        return handle_slice_sampler_exception(e, start_point, logprob, True)


def sample_hyperparameters(mcmc_iters, noiseless, input_points, func_values, cov_func, noise, amp2, ls, random_state=1):
    '''
    Samples hyper parameters for Gaussian processes.
    Args:
        mcmc_iters: the number of hyper-parameter samples required
        noiseless: the modeled function is noiseless
        input_points: all the points that have been evaluated so far
        func_values: the corresponding observed function values
        cov_func: the covariance function the Gaussian process uses
        noise: a starting value for the noise
        amp2: a starting value for the amplitude
        ls: an array of starting values for the length scales (size has to be the dimension of the input points)
    Returns:
        a list of hyper-parameter tuples
        the tuples are of the form (mean, noise, amplitude, [length-scales])
    '''
    np.random.seed(random_state)
    mean = np.mean(func_values)
    hyper_samples = []
    # sample hyper parameters
    for i in xrange(0, mcmc_iters ):
        if noiseless:
            noise = 1e-3
            [mean, amp2] = _sample_mean_amp_noise(input_points, func_values, cov_func, np.array([mean, amp2]), ls )
        else:
            [mean, amp2, noise] =_sample_mean_amp_noise(input_points, func_values, cov_func, np.array([mean, amp2, noise]), ls)
        ls = _sample_ls(input_points, func_values, cov_func, ls, mean, amp2, noise)
        #This is the order as expected
        log("mean: " + str(mean) + ", noise: " + str(noise) + " amp: " + str(amp2) + ", ls: " + str(ls))
        hyper_samples.append((mean, noise, amp2, ls))
    return hyper_samples
