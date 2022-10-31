import numpy as np
from helpers import *


def mean_squared_error_gd(y, tx, max_iters=150, gamma=0.005):
    """The Gradient Descent (GD) algorithm.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD 
    """
    # Define parameters to store w and loss
    loss = 0
    w = np.zeros((tx.shape[1],), dtype=float) #initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        
        w = w - gamma * gradient
    return w, loss


def mean_squared_error_sgd(y, tx, max_iters=150, gamma=0.005):
    """The Stochastic Gradient Descent algorithm (SGD).
            
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD 
    """
    # Define parameters to store w and loss
    loss = 0
    w = np.zeros((tx.shape[1],), dtype=float) #initial w
    batch_size = 1
    
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(y, tx, w)
            w = w - gamma * gradient
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    opt_weights = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    e = y - tx.dot(opt_weights)
    mse = 1/(2*len(y)) * e.T.dot(e)
    return opt_weights, mse


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar number
    """
    x_t = tx.T
    lambd = lambda_ * 2 * len(y)
    w = np.linalg.solve (np.dot(x_t, tx) + lambd * np.eye(tx.shape[1]), np.dot(x_t,y)) 
    loss = compute_mse(y, tx, w)
    return w,loss


def logistic_regression_gradient_descent(y, x):
    """calculates logistic regression model of given data set
    returns the weights and the loss

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar number
    """
    max_iter = 1000
    threshold = 1e-8
    gamma = .01
    losses = []

    # build tx
    tx = x
    w = np.zeros((tx.shape[1],), dtype=float)

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        # if iter % 300 == 0:
         #   print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    #print("loss={l}".format(l=calculate_loss_lr(y, tx, w)))
    
    return w, losses[-1]


def logistic_regression_regularized_gradient_descent(y, tx, lambda_, gamma=0.01):
    """calculates regularized logistic regression model of given data set, lamda, and gamma
    returns the weights and the loss

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        lambda_: scalar
        gamma: scalar

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar number
    """
    # init parameters
    max_iter = 1000
    threshold = 1e-8
    losses = []

    # build tx
    w = np.zeros((tx.shape[1],), dtype=float)

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        #if iter % 100 == 0:
         #   print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    #print("loss={l}".format(l=calculate_loss_lr(y, tx, w)))
    return w, losses[-1]