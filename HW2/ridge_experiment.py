'''
Created on 09-Feb-2015

@author: adarsh

Best way to use this would be to import this into a ipython notebook and try it out for yourself.
Most of this is template code that tests the other stuff out, mainly the shooting algo and projected SGD.

'''
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from shooting_algorithm import shooting_algorithm_vectorized, shooting_algorithm
from projected_sgd import projected_sgd
import time

# load the data
X = np.loadtxt("X.txt")
y = np.loadtxt("y.txt")


def ridge(X, y, Lambda):
    (N, D) = X.shape
    def ridge_obj(theta):
        return ((np.linalg.norm(np.dot(X, theta) - y)) ** 2) / (2 * N) + Lambda * (np.linalg.norm(theta)) ** 2
    return ridge_obj 

def compute_loss(X, y, theta):
    N = X.shape[0]
    return ((np.linalg.norm(np.dot(X, theta) - y)) ** 2) / (2 * N)


def best_lambda_ridge():
    """
    Simple linear search for best hyper parameter for the ridge objective
    """
    w = np.random.rand(X.shape[1], 1)
    X_train = X[:80, :]  # split into training and validation
    y_train = y[:80]
    X_val = X[80:100, :]
    y_val = y[80:100]
    w_min = w
    val_loss_min = float('inf')
    e = np.arange(-8, 1, 0.5)
    val_loss_hist = np.zeros(len(e))
    min_lambda = float('inf')
    for i in range(len(e)):
        Lambda = 10 ** e[i];
        w_opt = minimize(ridge(X_train, y_train, Lambda), w)
        val_loss = compute_loss(X_val, y_val, w_opt.x)
        val_loss_hist[i] = val_loss
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            min_lambda = Lambda
            w_min = w_opt.x
    
    print "Number of non zero coefficients(threshold=0.001):" + str(len(np.where(np.ravel((w_min) > 0.001))[0]))
    print "Number of non zero coefficients(threshold=0.01):" + str(len(np.where(np.ravel((w_min) > 0.01))[0]))
    print "Number of non zero coefficients(threshold=0.1):" + str(len(np.where(np.ravel((w_min) > 0.1))[0]))
    print "Min validation loss:" + str(val_loss_min)
    print "Min lambda:" + str(min_lambda)
    
    plt.plot(e, val_loss_hist, 'k', label="Validation loss for Ridge regression")
    plt.xlabel("log(lambda)")
    plt.ylabel("Validation loss")
    plt.show()

def best_lambda_shooting(X, y):
    """
    Simple linear search for best hyper parameter for the LASSO
    """
    l_s = np.arange(-4, 6, 0.25)
    
    min_loss = float('inf')
    w_min = None
    for l in l_s:
        w_shooting = shooting_algorithm(X[:80, :], y[:80], w_init=None, lambda_reg=10 ** l, num_iter=3000)
        loss = compute_loss(X[80:100, :], y[80:100], w_shooting)  # + str(w_shooting)
        if loss < min_loss:
            min_loss = loss
            w_min = w_shooting
    print "Min validation loss:" + str(min_loss)
    print "Number of non zero coefficients(threshold=0.01):" + str(len(np.where(((np.abs(w_min)) > 0.01))[0]))
    print w_min

def non_homotopy(X, y, num_iter=100):
    """
    Find best hyper parameter without warm start
    """
    X_train = X[:80, :]
    y_train = y[:80]
    X_val = X[80:100, :]
    y_val = y[80:100]
    
    
    lambda_max = 2 * np.max(np.abs(X_train.T.dot(y_train)))
    lambdas = np.zeros(num_iter)
    
    val_losses = np.zeros(num_iter)
    
    print lambda_max
    w_hist = np.zeros((num_iter, X.shape[1]))
    i = 0
    w = np.zeros((X.shape[1]))

    while i < num_iter :
        lambdas[i] = lambda_max

        val_losses[i] = compute_loss(X_val, y_val, w)
        w_hist[i] = w
        w = shooting_algorithm_vectorized(X_train, y_train, None, lambda_max, num_iter=2000)  # pass no initial theta
        lambda_max /= 2
                        
        i += 1
        
    print "Min lambda:" + str(lambdas[np.argmin(val_losses)])
    print "Min Loss:" + str(val_losses.min())
    print "Min w" + str(w_hist[np.argmin(val_losses)])
    plt.plot(lambdas, val_losses)
    plt.show()
    return lambdas[np.argmin(val_losses)], val_losses.min(), w_hist[np.argmin(val_losses)]


def homotopy(X, y, num_iter=100):
    """
    Homotopy as described in KPM chapter 13
    """
    X_train = X[:80, :]
    y_train = y[:80]
    X_val = X[80:100, :]
    y_val = y[80:100]
    
    
    lambda_max = 2 * np.max(np.abs(X_train.T.dot(y_train)))
    lambdas = np.zeros(num_iter)
    
    val_losses = np.zeros(num_iter)
    
    print lambda_max
    w_hist = np.zeros((num_iter, X.shape[1]))
    i = 0
    w = np.zeros((X.shape[1]))

    while i < num_iter :
        lambdas[i] = lambda_max

        val_losses[i] = compute_loss(X_val, y_val, w)
        w_hist[i] = w
        w = shooting_algorithm_vectorized(X_train, y_train, w, lambda_max, num_iter=2000)
        
        lambda_max /= 2
                        
        i += 1
        
    print "Min lambda:" + str(lambdas[np.argmin(val_losses)])
    print "Min Loss:" + str(val_losses.min())
    print "Min w" + str(w_hist[np.argmin(val_losses)])
    plt.plot(lambdas, val_losses)
    plt.show()
    return lambdas[np.argmin(val_losses)], val_losses.min(), w_hist[np.argmin(val_losses)]

def shooting_vs_sgd(X, y,num_iter = 100):
    """
    Compares the SGD with the shooting algo.
    """
    w = np.random.rand(X.shape[1], 1)
    
    X_train = X[:80, :]  # split into training and validation
    y_train = y[:80]
    X_val = X[80:100, :]
    y_val = y[80:100]
    
    w_shooting_min = w
    w_sgd_min = w
    
    val_shooting_min = float('inf')
    val_sgd_min = float('inf')
    
    lambda_max = 2 * np.max(np.abs(X_train.T.dot(y_train)))
    lambdas = np.zeros(num_iter)


    
    val_shooting_hist = np.zeros(num_iter)
    val_sgd_hist = np.zeros(num_iter)
    min_lambda = float('inf')
    
    w_shooting_opt = np.zeros(X_train.shape[1])
    w_sgd_opt = np.zeros(X_train.shape[1])
    i=0
    while i < num_iter :

        lambdas[i] = lambda_max

        Lambda = lambda_max;
        val_shooting_loss = compute_loss(X_val, y_val, w_shooting_opt)
        val_sgd_loss = compute_loss(X_val, y_val, w_sgd_opt)
        
        val_shooting_hist[i] = val_shooting_loss
        val_sgd_hist[i] = val_sgd_loss
        
        w_shooting_opt = shooting_algorithm_vectorized(X_train, y_train, None, Lambda)
        w_sgd_opt = projected_sgd(X_train, y_train, 0.005, Lambda, 1000)

       
        if val_shooting_loss < val_shooting_min:
            val_shooting_min = val_shooting_loss
            min_lambda = Lambda
            w_shooting_min = w_shooting_opt
        if val_sgd_loss < val_sgd_min:
            val_sgd_min = val_sgd_loss
            w_sgd_min = w_sgd_opt
        
        lambda_max /=2
        i+=1

    
    print "Min validation loss(shooting):" + str(val_shooting_min)
    print "Min validation loss(SGD):" + str(val_sgd_min)
    print "Min lambda:" + str(min_lambda)
    
    print w_shooting_min
    print w_sgd_min
    
    plt.plot(np.log(lambdas), val_shooting_hist, 'k', label="Validation loss for Shooting Algorithm")
    plt.plot(np.log(lambdas), val_sgd_hist, 'c', label="Validation loss for projected SGD Algorithm")
    plt.legend()
    plt.xlabel("log(lambda)")
    plt.ylabel("Validation loss")
    plt.show()


