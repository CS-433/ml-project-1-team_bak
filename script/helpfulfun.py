# -*- coding: utf-8 -*-
"""some helper functions."""

from cmath import isnan, nan
from re import A
import numpy as np
import copy

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

def compute_rmse(y, tx, w):
    mse = compute_mse(y, tx, w)
    return np.sqrt(2 * mse)

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

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
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
    return np.squeeze(- loss) * 1/len(y) 

def calculate_gradient_lr(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w).reshape((tx.shape[0],)))
    grad = tx.T.dot(pred - y) * 1/len(y) 
    return grad

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    pred = sigmoid(tx.dot(w).reshape((y.shape[0],)))
    pred = np.eye(len(y))*(pred.T[0])
    r = pred.dot(1-pred).reshape((y.shape[0],)) * 1/len(y)
    return tx.T.dot(r).dot(tx).reshape((w.shape[0],))

def learning_by_gradient_descent(y, tx, w, gamma):
    """one step of gradient descent using logistic regression. Return the loss and the updated w"""
    loss= calculate_loss_lr(y, tx, w)
    w = w - gamma * calculate_gradient_lr(y, tx, w)
    return loss, w

def logistic_regression(y, tx, w):
    """return the loss, gradient of the loss, and hessian of the loss."""
    loss = calculate_loss_lr(y,tx,w)
    grad = calculate_gradient_lr(y, tx, w)
    hess = calculate_hessian(y,tx,w)
    return (loss, grad, hess)
    
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
    
    return loss_tr, 

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    # Generate the grid of parameters to be swept
    
    opt_weights = np.linalg.solve(tx.T.dot(tx),(tx.T.dot(y)))

    e = y - tx.dot(opt_weights)
    mse = 1/(2*len(y)) * e.T.dot(e)
    return opt_weights, mse

def get_median(a):
    a[a == -999.0] = np.nan
    median=np.nanmedian(a)
    return median

#def replace_nans_with_median(a,median):
 #   a[isnan(a)] = median
    
    
def replace_with_median(array_of_features):
    replaced=copy.deepcopy(array_of_features)
    for i in range(replaced[0,:].size):
        #when median is directly computed the value is -999 for columns with alot of missing data
        #find median while dismissing -999 values
        median_value=get_median(replaced[:,i])
        #replace_nans_with_median(array_of_features[:,i],median_value)
        for j in range(replaced[:,i].size):
            if np.isnan(replaced[j,i]):
                replaced[j,i] = median_value
    return replaced


def colinearity_check(array_of_features,critical_value):
    #regress each feature to all other features
        #write the code such that it drops same pairs
    #calculate R^2 value
    #Use the VIF method 1/(1-R^2) - this will give the colinearity check value
    #how to find R^2
    #run machine learning for linear regression
    #use MSE and use least squares to find the regression
    #R^2 in terms of MSE and Var(y)
    #print out the colinearity values of feature pairs
    features_VIF=np.array(["features x-y","VIF value"])
    
    
    for i in range(array_of_features[0,:].size):
        y=array_of_features[:,i]
        #print(y,y.shape)
        for j in range(i+1,array_of_features[0,:].size):
            #print(j)
            num_samples = len(y)
            tx = np.c_[np.ones(num_samples), array_of_features[:,j]]
            #print(tx,tx.shape)
            pt_w,mse=least_squares(y,tx)
            R_square=1-mse/np.var(y)
            VIF=1/(1-R_square)
            if VIF >= critical_value:
                features_VIF=np.append(features_VIF, [(str(i+1)+"-"+str(j+1)),VIF])
                
    #Does this make sense
    #Or can we just use the panda library and say that these are linear - does it count as being a part of our code!
    #As it is not excatly within the algorithm

    #TO DO - Feature Selection
    #variance - covariance - heat plot() -- look at this at first -pairs plot
    #correlation equation
    #just use pandas and write in the report as if we didnt use
    #outliers
    #unbalanced data - what to do
    #replace missing with median and outliers with median
    #spitting data set data
    #removed old commits - answers
    return features_VIF
