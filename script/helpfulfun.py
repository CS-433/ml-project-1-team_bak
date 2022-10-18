# -*- coding: utf-8 -*-
"""some helper functions."""

import numpy as np

# COMPUTING LOSS 
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
    pred = sigmoid(tx.dot(w).reshape((tx.shape[0],)))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def calculate_gradient_lr(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w).reshape((tx.shape[0],)))
    grad = tx.T.dot(pred - y)
    return grad

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    pred = sigmoid(tx.dot(w).reshape((tx.shape[0],)))
    pred = np.diag(pred.T[0])
    r = np.multiply(pred, (1-pred))
    return tx.T.dot(r).dot(tx)
