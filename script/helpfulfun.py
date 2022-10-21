# -*- coding: utf-8 -*-
"""some helper functions."""

import numpy as np

# Linear Regression
#*************************************************************************
def compute_error(y,tx,w):
    return y - tx.dot(w).reshape((y.shape[0],))
    
def compute_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def compute_mae(e):
    """Calculate the mae for vector e."""
    return np.abs(e).mean()

# by default MSE
def compute_loss(y, tx, w):
    """Calculate the loss using either MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = compute_error(y, tx, w)
    return compute_mse(e)

def compute_loss_mae(y, tx, w):
    
    e = compute_error(y, tx, w)
    return compute_mae(e)

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    matrix = np.zeros((x.shape[0], degree + 1))
    for i in range(x.shape[0]):
        for j in range(degree + 1):
            matrix[i][j] = x[i]**j
    return matrix

def polynomial_regression():
    """Constructing the polynomial basis function expansion of the data,
       and then running least squares regression."""
    # define parameters
    degrees = [1, 3, 7, 12]
    
    # define the structure of the figure
    num_row = 2
    num_col = 2
    f, axs = plt.subplots(num_row, num_col)

    for ind, degree in enumerate(degrees):
        matrix_x_poly = build_poly(x, degree)
        weights, e = least_squares(y, matrix_x_poly)
        rmse = np.sqrt(2*compute_mse(e))
        
        print("Processing {i}th experiment, degree={d}, rmse={loss}".format(i=ind + 1, d=degree, loss=rmse))
        # plot fit
        plot_fitted_curve(y, x, weights, degree, axs[ind // num_col][ind % num_col])
        plt.tight_layout()

    plt.show()

# COMPUTING GRADIENT 
#*************************************************************************
def compute_gradient(y, tx, w):
    """Computes the gradient at w.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = compute_error(y, tx, w)
    grad = (-1/y.shape[0]) * (tx.T.dot(e))
    return grad

# Preprocessing
#*************************************************************************
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    tx = np.c_[np.ones(x.shape[0]), x]
    return tx, mean_x, std_x

# LOGISTIC REGRESSION FUNCTIONS
#*************************************************************************
def sigmoid(t):
    # sigmoid function is applied on t.
    return 1.0 / (1 + np.exp(-t))

def calculate_loss_lr(y, tx, w):
    # computing the cost by negative log likelihood.
    #print(y.shape[0], tx.shape[0])
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    
    pred = sigmoid(tx.dot(w).reshape((y.shape[0],)))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)*1/len(y) 

def calculate_gradient_lr(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w).reshape((y.shape[0],)))
    grad = 1/len(y) * tx.T.dot(pred - y)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """one step of gradient descent using logistic regression. Return the loss and the updated w"""
    loss= calculate_loss_lr(y, tx, w)
    w = w - gamma * calculate_gradient_lr(y, tx, w)
    return loss, w

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    pred = sigmoid(tx.dot(w).reshape((y.shape[0],)))
    pred = np.eye(len(y))*(pred.T[0])
    r = pred.dot(1-pred).reshape((y.shape[0],)) * 1/len(y)
    return tx.T.dot(r).dot(tx).reshape((w.shape[0],))

def logistic_regression(y, tx, w):
    """return the loss, gradient of the loss, and hessian of the loss."""
    loss = calculate_loss_lr(y,tx,w)
    grad = calculate_gradient_lr(y, tx, w)
    hess = calculate_hessian(y,tx,w)
    return (loss, grad, hess)
    #raise NotImplementedError
    
def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""
    pred = sigmoid(tx.dot(w).reshape((tx.shape[0],))) 
    loss, grad, hess = logistic_regression(y, tx, w)
    loss = loss +  lambda_ * w.T.dot(w).reshape((1,))
    grad = grad + 2*lambda_ * w
    return loss, grad 

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w."""
    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * grad
    return loss, w

def learning_by_newton_method(y, tx, w, gamma):
    """Do one step of Newton's method. Return the loss and updated w."""
    loss, grad, hess = logistic_regression(y, tx, w)
    w_init = w
    w = w_init - gamma * np.linalg.solve(hess,grad)
    return loss, w


# CROSS VALIDATION
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for a fold corresponding to k_indices"""
    x_test_sub = x[k_indices[k]]
    x_train_sub = []
    for i in range(k_indices.shape[0]):
        if i != k:
            x_train_sub = np.concatenate((x_train_sub, x[k_indices[i]]), axis=None)       
    y_test_sub = y[k_indices[k]]
    y_train_sub = []
    for i in range(k_indices.shape[0]):
        if i != k:
            y_train_sub = np.concatenate((y_train_sub, y[k_indices[i]]), axis=None)
    
    tx_train  = build_poly(x_train_sub, degree)
    tx_test = build_poly(x_test_sub, degree)
    
    w = ridge_regression(y_train_sub, tx_train, lambda_)
    
    loss_tr = np.sqrt(2 * compute_mse(y_train_sub, tx_train, w))
    loss_te = np.sqrt(2 * compute_mse(y_test_sub, tx_test, w))
    
    return loss_tr, loss_te





