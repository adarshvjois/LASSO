'''
Created on 12-Feb-2015

@author: adarsh
Implemented with reference to Bottou's SGD tricks.
'''
import numpy as np

def projected_sgd(X, y, alpha=0.001, lambda_reg=1, num_iter=1000):
    """
    X - Training set
    y - training values to classify/do regression on.
    alpha - step size for SGD
    lambda_reg - hyper parameter
    num_iter - iterations for SGD
    
    returns the final weight vector.
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    u_i = np.ones(num_features)  # Initialize theta
    v_i = np.ones(num_features)
    
    theta_hist = np.zeros((num_iter, num_instances, num_features))  # Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances))  # Initialize loss_hist
    # TODO
    if isinstance(alpha, str):
        if alpha == '1/t':
            f = lambda x: 1.0 / x
        elif alpha == '1/sqrt(t)':
            f = lambda x: 1.0 / np.sqrt(x)
        alpha = 0.01
    elif isinstance(alpha, float):
        f = lambda x: 1
    else:
        return
    
    for t in range(num_iter):
        
        for i in range(num_instances):
            gamma_t = alpha * f((i + 1) * (t + 1))
            # the gradient thats been split into two
            theta_hist[t , i] = u_i - v_i
            # compute loss for current theta
            loss = y[i] - np.dot(X[i], u_i - v_i)
            # reg. term
            regulariztion_loss = lambda_reg * np.sum(u_i + v_i)
            # for gradient
            regularization_penalty = lambda_reg 
            # gradent calc.
            grad_plus = regularization_penalty - X[i] * (loss)
            u_i = u_i - gamma_t * grad_plus
            
            u_i[u_i < 0] = 0
            
            grad_minus = X[i] * (loss) + regularization_penalty
            v_i = v_i - gamma_t * grad_minus
            v_i[v_i < 0] = 0
            # squared loss
            loss_hist[t, i] = (loss) ** 2 + regulariztion_loss 
                                    
    return u_i - v_i
