'''
Created on Oct 28, 2013

@author: Simon Bartels, Aaron Klein
'''

import spearmint.gp as gp
import numpy as np
np.random.seed(1)

import scipy.linalg as spla
import copy

def fetchKernel(covar_name):
    '''
    Returns the corresponding function and gradient function for the given covariance function name.
    Searches the spearmint gp class and this one.
    Args:
        covar_name: the name of the covariance function (String)
    Returns:
        a tuple of two functions: the covariance function and the gradient function
    '''
    try:
        f = getattr(gp, covar_name)
        df = getattr(gp, 'grad_' + covar_name)
        return (f,df)
    except:
        f = globals()[covar_name]
        df = globals()['grad_' + covar_name]
        return (f, df)
    
    
def getNumberOfParameters(covarname, input_dimension):
    '''
    Returns the number of parameters the kernel has.
    Args:
        covarname: the name of the covariance function
        input_dimension: dimensionality of the input arguments
    Returns:
        the number of parameters
    '''
    try:
        #try to find covariance function in spearmint GP class
        getattr(gp, covarname)
        #then it is just the input dimension
        return input_dimension
    except:
        if covarname == 'Polynomial3':
            return 1
        elif covarname == 'Linear':
            return 1
        elif covarname == 'LogLinear':
            return getNumberOfParameters('Linear', input_dimension)
        elif covarname == 'Normalized_Polynomial3':
            return getNumberOfParameters('Polynomial3', input_dimension)
        elif covarname == 'BigData':
            return getNumberOfParameters('Polynomial3', 1)+getNumberOfParameters('Matern52', input_dimension-1)
        elif covarname == 'CostKernel':
            return getNumberOfParameters('Polynomial3', 1)+getNumberOfParameters('Matern52', input_dimension-1)
        elif covarname == 'CostKernel2':
            return getNumberOfParameters('Polynomial3', input_dimension)+getNumberOfParameters('Matern52', input_dimension)
        else:
            raise NotImplementedError('The given covariance function (' + covarname + ') was not found.')
        
        
def _polynomial3_raw(ls, x1, x2=None, value=True, grad=False):
    factor = 1
    if x2 is None:
        #in this case k(x,y) has to be considered of the form k(x)
        #and dk/dx is then 3(x^Tx+c)^2*(2x)
        factor = 2
        x2=x1
    #we may assume the input is a matrix of the form N x D
    #c = np.empty([x1.shape[0],x2.shape[0]])
    #c.fill(ls[0])
    c = ls[0]
    dot = np.dot(x1, x2.T) + c
    if grad:
        #compute dk(x1,x2)/dx2
        #=3 * apply(dot, ^2) o x1
        #dk = factor * 3 * (dot ** 2) * x1
        #this is to get the gradient in spearmint's format
        dk = np.array([(factor * 3 * (dot[i] ** 2)) * np.array([x1[i]]) for i in range(0, x1.shape[0])])
        
        #because the spearmint implementations all invert the signs of their gradients we will do so as well
        dk = -dk        
        if value:
            #we want both: value and gradient
            k = dot ** 3
            return (k, dk)
        #we want just the gradient
        return dk
    else:
        #we want just the kernel value
        k = dot ** 3
        return k

#TODO: generalize polynomial kernel???
def Linear(ls, x1, x2=None, grad=False):
    factor = 1
    if x2 is None:
        factor = 2
        x2=x1
    c = ls[0]
    dot = np.dot(x1, x2.T) + c
    if grad:
        dk = np.array([factor * np.array([x1[i]]) for i in range(0, x1.shape[0])])
        #because the spearmint implementations all invert the signs of their gradients we will do so as well
        dk = -dk
        return (dot, dk)
    return dot

def grad_Linear(ls, x1, x2=None):
    factor = 1
    if x2 is None:
        factor = 2
    dk = np.array([factor * np.array([x1[i]]) for i in range(0, x1.shape[0])])
    #because the spearmint implementations all invert the signs of their gradients we will do so as well
    dk = -dk
    return dk

#TODO: kernel has sharp edge close to zero. Is there something we can do about it?
def LogLinear(ls, x1, x2=None, grad=False):
    '''
    Linear kernel that puts the inputs on a log scale.
    '''
    if x2 is not None:
        return Linear(ls, np.log(x1+1e-15), np.log(x2+1e-15), grad)
    return Linear(ls, np.log(x1+1e-15), None, grad)

def grad_LogLinear(ls, x1, x2=None):
    if x2 is not None:
        return grad_Linear(ls, np.log(x1+1e-15), np.log(x2+1e-15))
    return grad_Linear(ls, np.log(x1+1e-15), None)

def grad_Polynomial3(ls, x1, x2=None):
    return _polynomial3_raw(ls, x1, x2, value=False, grad=True)

def Polynomial3(ls, x1, x2=None, grad=False):
    return _polynomial3_raw(ls, x1, x2, True, grad)

def Normalized_Polynomial3(ls, x1, x2=None, grad=False):
    #TODO: quick and dirty, refactor!
    compute_grad = False
    if grad:
        (k, dk) = _polynomial3_raw(ls, x1, x2, True, True)
        compute_grad = True
    else:
        k = _polynomial3_raw(ls, x1, x2)
    if x2 is None:
        x2 = x1
        if grad:
            dk = np.array([np.array([np.zeros(x1.shape[1])])])
            compute_grad = False
    for i in range(0, x1.shape[0]):
        if compute_grad:
            #we assume x2 was not none!!!
            sqrt_kxx = np.sqrt(_polynomial3_raw(ls, np.array(x1[i])))
            sqrt_kyy = np.sqrt(_polynomial3_raw(ls, x2))
            kxy = k[i]
            dk[i][0] = 1/sqrt_kxx*(dk[i][0]/sqrt_kyy-0.5*kxy*grad_Polynomial3(ls, x2)[0]/(sqrt_kyy**3))
        for j in range(0, x2.shape[0]):
            ki = _polynomial3_raw(ls, np.array(x1[i]))
            kj = _polynomial3_raw(ls, np.array(x2[j]))
            k[i][j] = k[i][j]/(np.sqrt(ki)*np.sqrt(kj))
    if not grad:
        return k
    return (k,dk)

def grad_Normalized_Polynomial3(ls, x1, x2=None):
    #TODO: quick and dirty, refactor!
    (k, dk) = _polynomial3_raw(ls, x1, x2, True, True)
    if x2 is None:
        return np.array([np.array([np.zeros(x1.shape[1])])])
    for i in range(0, x1.shape[0]):
        #we assume x2 was not none!!!
        sqrt_kxx = np.sqrt(_polynomial3_raw(ls, np.array(x1[i])))
        sqrt_kyy = np.sqrt(_polynomial3_raw(ls, x2))
        kxy = k[i]
        dk[i][0] = 1/sqrt_kxx*(dk[i][0]/sqrt_kyy-0.5*kxy*grad_Polynomial3(ls, x2)[0]/(sqrt_kyy**3))
    return dk

def _sum_kernel_raw(kf1, kf2, ls1, ls2, x1, x2=None, value=True, grad=False):
    '''
    Sum of kernels, i.e. kernel is of the form k(x,y)=k1(x,y)+k2(x,y).
    Args:
        kf1: the first kernel function (with standard Spearmint inputs, 
            i.e. CAN NOT be e.g. this kernel)
        kf2: the second kernel function (of standard Spearmint inputs)
        ls1: length scales for first kernel
        ls2: length scales for second kernel
        x11: first argument for both kernels
        x22: second argument for both kernels
        value: compute value of the kernel
        grad: compute gradient of the kernel
    Returns:
        either tuple (value, gradient) or single element
    '''
    if not grad:
        #only the value is of interest
        k1 = kf1(ls1, x1, x2)
        k2 = kf2(ls2, x1, x2)
        k = k1+k2
        return k
    else:
        (k1, dk1) = kf1(ls1, x1, x2, True)
        (k2, dk2) = kf2(ls2, x1, x2, True)
        #sum rule
        dk = dk1 + dk2
        if not value:
            #we care only for the gradient
            return dk
        k = k1+k2
        return (k,dk)
    
def _product_kernel_raw(kf1, kf2, ls1, ls2, x1, x2=None, value=True, grad=False):
    '''
    Sum of kernels, i.e. kernel is of the form k(x,y)=k1(x,y)+k2(x,y).
    Args:
        kf1: the first kernel function (with standard Spearmint inputs, 
            i.e. CAN NOT be e.g. this kernel)
        kf2: the second kernel function (of standard Spearmint inputs)
        ls1: length scales for first kernel
        ls2: length scales for second kernel
        x11: first argument for both kernels
        x22: second argument for both kernels
        value: compute value of the kernel
        grad: compute gradient of the kernel
    Returns:
        either tuple (value, gradient) or single element
    '''
    if not grad:
        #only the value is of interest
        k1 = kf1(ls1, x1, x2)
        k2 = kf2(ls2, x1, x2)
        k = k1*k2
        return k
    else:
        (k1, dk1) = kf1(ls1, x1, x2, True)
        (k2, dk2) = kf2(ls2, x1, x2, True)
        #product rule
        dk = dk1*k2 + k1*dk2
        if not value:
            #we care only for the gradient
            return dk
        k = k1*k2
        return (k,dk)

def _sum_kernel_raw_d(kf1, kf2, ls1, ls2, x11, x12, x21=None, x22=None, value=True, grad=False):
    '''
    Sum of kernels with disjunct inputs. I.e. kernel is of the form:
    k((x1,x2),(y1,y2))=k1(x1,y1)+k2(x2,y2)
    Args:
        kf1: the first kernel function (with standard Spearmint inputs, 
            i.e. CAN NOT be e.g. this kernel)
        kf2: the second kernel function (of standard Spearmint inputs)
        ls1: length scales for first kernel
        ls2: length scales for second kernel
        x11: first argument for first kernel
        x12: first argument for second kernel
        x21: second argument for first kernel
        x22: second argument for second kernel
        value: compute value of the kernel
        grad: compute gradient of the kernel
    Returns:
        either tuple (value, gradient) or single element
    '''
    num_of_elements = x11.shape[0]
    if not grad:
        #only the value is of interest
        k1 = kf1(ls1, x11, x21)
        k2 = kf2(ls2, x12, x22)
        k = np.array([k1[i]+k2[i] for i in range(0, num_of_elements)])
        return k
    else:
        (k1, dk1) = kf1(ls1, x11, x21, True)
        (k2, dk2) = kf2(ls2, x12, x22, True)
        #sum rule
        #dk = dk1+dk2
        dk = np.array([np.concatenate((dk1[i], dk2[i]), axis=1) 
                       for i in range(0, num_of_elements)])
        if not value:
            #we care only for the gradient
            return dk
        k = np.array([k1[i]+k2[i] for i in range(0, num_of_elements)])
        return (k,dk)
    
def _product_kernel_raw_d(kf1, kf2, ls1, ls2, x11, x12, x21=None, x22=None, value=True, grad=False):
    '''
    Product of kernels with disjunct inputs. I.e. kernel is of the form:
    k((x1,x2),(y1,y2))=k1(x1,y1)*k2(x2,y2)
    Args:
        kf1: the first kernel function (with standard Spearmint inputs, 
            i.e. CAN NOT be e.g. this kernel)
        kf2: the second kernel function (of standard Spearmint inputs)
        ls1: length scales for first kernel
        ls2: length scales for second kernel
        x11: first argument for first kernel
        x12: first argument for second kernel
        x21: second argument for first kernel
        x22: second argument for second kernel
        value: compute value of the kernel
        grad: compute gradient of the kernel
    Returns:
        either tuple (value, gradient) or single element 
    '''
    num_of_elements = x11.shape[0]
    if not grad:
        #only the value is of interest
        k1 = kf1(ls1, x11, x21)
        k2 = kf2(ls2, x12, x22)
        k = np.array([k1[i]*k2[i] for i in range(0, num_of_elements)])
        return k
    else:
        (k1, dk1) = kf1(ls1, x11, x21, True)
        (k2, dk2) = kf2(ls2, x12, x22, True)
        #product rule
        dk = np.array([np.concatenate((dk1[i]*k2[i], k1[i]*dk2[i]), axis=1) 
                       for i in range(0, num_of_elements)])
        if not value:
            #we care only for the gradient
            return dk
        k = np.array([k1[i]*k2[i] for i in range(0, num_of_elements)])
        return (k,dk)
    
def _CostKernel2_raw(ls, x1, x2=None, value=True, grad=False):
    p = getNumberOfParameters('Polynomial3', x1.shape[1])
    ls1 = ls[:p]
    ls2 = ls[p:]
    return _sum_kernel_raw(Polynomial3, gp.Matern52, ls1, ls2, x1, x2, value, grad)

def CostKernel2(ls, x1, x2=None, grad=False):
    return _CostKernel2_raw(ls, x1, x2, grad)

def grad_CostKernel2(ls, x1, x2=None):
    return _CostKernel2_raw(ls, x1, x2, False, True)

def _CostKernel3_raw(ls, x1, x2=None, value=True, grad=False):
    p = getNumberOfParameters('Polynomial3', x1.shape[1])
    ls1 = ls[:p]
    ls2 = ls[p:]
    return _product_kernel_raw(Polynomial3, gp.Matern52, ls1, ls2, x1, x2, value, grad)

def CostKernel3(ls, x1, x2=None, grad=False):
    return _CostKernel3_raw(ls, x1, x2, grad)

def grad_CostKernel3(ls, x1, x2=None):
    return _CostKernel3_raw(ls, x1, x2, False, True)
    
def CostKernel(ls, x1, x2=None, grad=False):
    ls1 = ls[:1]
    ls2 = ls[1:]
    x11 = x1[:,:1]
    x12 = x1[:,1:]
    x21 = None
    x22 = None
    if x2 is not None:
        x21 = x2[:,:1]
        x22 = x2[:,1:]
    return _sum_kernel_raw_d(Polynomial3, gp.Matern52, ls1, ls2, x11, x12, x21, x22, True, grad)

def grad_CostKernel(ls, x1, x2=None):
    ls1 = ls[:1]
    ls2 = ls[1:]
    x11 = x1[:,:1]
    x12 = x1[:,1:]
    x21 = None
    x22 = None
    if x2 is not None:
        x21 = x2[:,:1]
        x22 = x2[:,1:]
    return _sum_kernel_raw_d(Polynomial3, gp.Matern52, ls1, ls2, x11, x12, x21, x22, False, True)

def _bigData_raw(ls, x1, x2=None, value=True, grad=False):
    x11 = x1[:,:1] #get first entry of each vector
    x12 = x1[:,1:] #get the rest
    x21 = None
    x22 = None
    if not(x2 is None):
        x21 = x2[:,:1]
        x22 = x2[:,1:]
    ls1 = ls[:1]
    ls2 = ls[1:]
    return _product_kernel_raw_d(LogLinear, gp.Matern52, ls1, ls2, x11, x12, x21, x22, value, grad)

def grad_BigData(ls,x1,x2=None):
    return _bigData_raw(ls, x1, x2, value=False, grad=True)

def BigData(ls, x1, x2=None, grad=False):
    return _bigData_raw(ls, x1, x2, True, grad)
        


class GPModel(object):

    def __init__(self, X, y, mean, noise, amp2, ls, covarname="Matern52", cholesky=None, alpha=None, cov_func = None, covar_derivative=None):
        '''
            Constructor
            Args:
            X: The observed inputs.
            y: The corresponding observed values.
            mean: the mean value
            noise: the noise value
            amp: the amplitude
            ls: the length scales (numpy array of length equialent to the dimension of the input points)
            covarname: the name of the covariance function (see spearmint class gp)
            
            The following arguments are only used internally for the copy() method.
            cholesky: if already available the cholesky of the kernel matrix
            alpha: if cholesky is passed this one needs to be set, too. It's L/L/(y-mean).
        '''
        
        self._X = X
        self._y = y
        self._ls = ls
        self._amp2 = amp2
        self._mean = mean
        self._noise = noise
        if cholesky is None:
            self._cov_func, self._covar_derivative = fetchKernel(covarname)
            self._compute_cholesky()
        else:
            self._cov_func = cov_func
            self._covar_derivative =covar_derivative
            self._L = cholesky
            self._alpha = alpha

    def getValues(self):
        return self._y

    def getPoints(self):
        return self._X

    def _compute_cholesky(self):
        #the Cholesky of the correlation matrix
        K = self._compute_covariance(self._X) + self._noise * np.eye(self._X.shape[0])
        self._L = spla.cholesky(K, lower=True)
        self._alpha = spla.cho_solve((self._L, True), self._y - self._mean)

    def predict(self, Xstar, variance=False):
        kXstar = self._compute_covariance(self._X, Xstar)
        func_m = np.dot(kXstar.T, self._alpha) + self._mean
        if not variance:
            return func_m

        beta = spla.solve_triangular(self._L, kXstar, lower=True)

        #old spearmint line - their kernels have k(x,x)=1
        #func_v = self._amp2 * (1 + 1e-6) - np.sum(beta ** 2, axis=0)

        #prior variance is basically diag(k(X,X))
        prior_variance = np.empty(Xstar.shape[0])
        for i in range(0, Xstar.shape[0]):
            prior_variance[i] = self._compute_covariance(np.array([Xstar[i]]))[0]
        func_v = prior_variance - np.sum(beta ** 2, axis=0) #np.dot(beta.T, beta)
        return (func_m, func_v)

    def predict_vector(self, input_point):
        (func_m, func_v) = self.predict(np.array([input_point]), True)
        return (func_m[0], func_v[0])

    def _compute_covariance(self, x1, x2=None):
        if x2 is None:
            return self._amp2 * (self._cov_func(self._ls, x1, None)
                                + 1e-6 * np.eye(x1.shape[0]))
        else:
            return self._amp2 * self._cov_func(self._ls, x1, x2)

    def getGradients(self, xstar):
        xstar = np.array([xstar])
        # gradient of mean
        #This is what Andrew McHutchon in "Differentiating Gaussian Processes"
        #proposes. 
        #dk = self._amp2 * self._covar_derivative(self._ls, xstar, self._X)[0]
        #grad_m = np.dot(dk.T, self._alpha)
        #The sign of the version above agrees with the first order approximation.
        #The spearmint implementation not.
        #THEREFORE WE TURN THE SIGN OF DK! BEWARE: Also used for grad_v!!!
        # Below is the code how spear mint does it.
        dk = -self._amp2 * self._covar_derivative(self._ls, self._X, xstar)
        dk = np.squeeze(dk)
        grad_m = np.dot(self._alpha.T, dk)
        
        # gradient of variance    
        #intuitively k should be cov(xstar,X) but that gives a row vector!
        k = self._compute_covariance(self._X, xstar)
        #kK = k^T * K^-1
        kK = spla.cho_solve((self._L, True), k)
        #s'(x)=-dk^T kK /s(x). So for the derivative of v(x) in terms of s(x) we have:
        #v(x)=s^2(x) <=> v'(x)=2s(x)*s'(x)
        grad_v = -2 * np.dot(kK.T, dk)
        return (grad_m, grad_v)

    def getNoise(self):
        return self._noise
    
    def getAmplitude(self):
        return self._amp2
        
    def sample(self, x, omega):
        '''
            Gives a stochastic prediction at x
            Parameters In: omega = sample from standard normal distribution 
                            x = the prediction point (vector)
            Parameters Out: y = function value at x
        '''
        x = np.array([x])
        cholsolve = spla.cho_solve((self._L, True), self._compute_covariance(self._X, x))
        cholesky = np.sqrt(self._compute_covariance(x, x) - 
                           np.dot(self._compute_covariance(self._X, x).T, cholsolve)+self._noise)
        y = self.predict(x,False) + cholesky * omega
        #y is in the form [[value]] (1x1 matrix)
        return y[0][0]
        
    def update(self, x, y):
        '''
            Adds x,y to the observation and creates a new GP 
            Args:
                x: a numpy vector
                y: a single value
        '''
        self._X = np.append(self._X, np.array([x]), axis=0)
        self._y = np.append(self._y, np.array([y]), axis=0)
        #TODO: Use factor update
        self._compute_cholesky()

    def getCholeskyForJointSample(self, Xstar):
        '''
        Computes the cholesky decomposition of the covariance matrix and the mean prediction at Xstar.
        Returns:
            (mean, cholesky)
        '''
        kXstar = self._compute_covariance(self._X, Xstar)
        cholsolve = spla.cho_solve((self._L, True), kXstar)
        Sigma = (self._compute_covariance(Xstar, Xstar) -
                  np.dot(kXstar.T, cholsolve))
        cholesky = spla.cholesky(Sigma + self._noise*np.eye(Sigma.shape[0]),lower=True)
        return (self.predict(Xstar,False), cholesky)
        
    def drawJointSample(self, mean, L, omega):
        '''
            Draws a joint sample for mean and cholesky decomposition as computed with #getCholeskyForJointSample
            Args:
                mean: the first return value of #getCholeskyForJointSample
                L: the second return value of #getCholeskyForJointSample
                omega: a vector of samples from the standard normal distribution, one for each point
            Returns:
                a numpy array (vector)
        '''
        #the computation follows "Entropy Search for Information-Efficient Global Optimization"
        # by Hennig and Schuler
        y = mean + np.dot(L, omega)
        return y
    
    def copy(self):
        '''
        Returns a copy of this object.
        Returns:
            a copy
        '''
        X = copy.copy(self._X)
        y = copy.copy(self._y)
        ls = copy.copy(self._ls)
        L = copy.copy(self._L)
        alpha = self._alpha
        return GPModel(X, y, self._mean, self._noise, self._amp2, ls, None, L, alpha, self._cov_func, self._covar_derivative)