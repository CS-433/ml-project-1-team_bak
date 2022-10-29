import numpy as np


# COMPUTING LOSS 
#*************************************************************************
def compute_error(y,tx,w):
    return y - tx.dot(w).reshape((y.shape[0],))
    

def compute_mse(y, tx, w):
    """
    Compute the Mean Square Error as defined in class.
    Takes as input the targeted y, the sample matrix X and the feature fector w.
    """
    e = y - tx@w
    mse = e.T.dot(e) /(2*len(e))
    return mse


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
    return compute_mse(y, tx, w)


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


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""
    loss = calculate_loss_lr(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = calculate_gradient_lr(y, tx, w) + 2 * lambda_ * w*0
    return loss, gradient


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w =  w - gamma * gradient
    return loss, w


def learning_by_gradient_descent(y, tx, w, gamma):
    """one step of gradient descent using logistic regression. Return the loss and the updated w"""
    loss= calculate_loss_lr(y, tx, w)
    w = w - gamma * calculate_gradient_lr(y, tx, w)
    return loss, w


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
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

            
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0] 
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

#########################################################################
#Predictions#
#########################################################################

def get_predictions(weights, data, log=False):
    """get prediction results in binary format"""
    y_pred = np.dot(data, weights)
    if log:
        y_pred[np.where(y_pred <= 0.5)] = 0
        y_pred[np.where(y_pred > 0.5)] = 1
    else:
        y_pred[np.where(y_pred <= 0)] = -1
        y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def compute_accuracy(y_pred, y):
    """Computes accuracy"""
    #o = np.subtract(y,y_pred)
    #true = o[o==0]
    sum = 0
    for idx, y_val in enumerate(y):
        if y_val == y_pred[idx]:
            sum += 1
    return sum / len(y)
