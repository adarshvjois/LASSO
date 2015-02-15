'''
Created on 10-Feb-2015

@author: adarsh
References: KPM and KPM's MATLAB vectorized code
@https://code.google.com/p/pmtk3/source/browse/trunk/toolbox/Variable_selection/lassoExtra/LassoShooting.m?r=1393
'''

import numpy as np

def shooting_algorithm(X, y, w_init=None, lambda_reg=0.1, num_iter=1000):
    """
    X - Training set
    y - training values to classify/do regression on.
    w_init - my initial guess for the weights
    lambda_reg - hyper parameter
    
    returns the final weight vector.
    """
   
    if w_init is not None:
        w = w_init 
    else:
        w = np.ones(X.shape[1])
        
    D = X.shape[1]
    i = 0
    
    while  i < num_iter:  
        i += 1
        for j in range(D):
            X_j = X[:, j]
            a_j = 2 * X_j.T.dot(X_j)
            c_j = 2 * X_j.T.dot(y - X.dot(w) + w[j] * X_j)
            if c_j < -lambda_reg:
                w[j] = (c_j + lambda_reg) / a_j
            elif c_j > lambda_reg:
                w[j] = (c_j - lambda_reg) / a_j
            else:
                w[j] = 0
    return w

def shooting_algorithm_vectorized(X, y, w_init=None , lambda_reg=0.1, num_iter=1000):
    """
    X - Training set
    y - training values to classify/do regression on.
    w_init - my initial guess for the weights
    lambda_reg - hyper parameter
    
    returns the final weight vector.
    """
    #TODO add a way in which I can include tolerance.
    if w_init is not None:
        w = w_init 
    else:
        w = np.ones(X.shape[1])
    D = X.shape[1]

    XX2 = np.dot(X.T, X) * 2
    Xy2 = np.dot(X.T, y) * 2
    i = 0
    while i < num_iter:
        i += 1
        for j in range(D):
            c_j = Xy2[j] - XX2[j, :].dot(w) + XX2[j, j] * w[j]
            a_j = XX2[j, j]
            
            if c_j < -lambda_reg:
                w[j] = (c_j + lambda_reg) / a_j
            elif c_j > lambda_reg:
                w[j] = (c_j - lambda_reg) / a_j
            else:
                w[j] = 0
    return w
