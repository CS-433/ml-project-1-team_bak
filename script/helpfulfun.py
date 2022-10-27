# -*- coding: utf-8 -*-
"""some helper functions."""

from cmath import isnan, nan
from re import A
import numpy as np
import copy
import matplotlib.pyplot as plt

########################################################################
########### NEW FUNCTIONs THAT I HAVE INTRODUCED ########################
########################################################################

def load_csv_data(data, sub_sample=False):
    """Loads data and returns y (class labels), x (features) and indices (event ids)"""
    y = np.genfromtxt(data, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data, delimiter=",", skip_header=1)
    indices = x[:, 0].astype(np.int)
    x = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    y_num = np.ones(len(y))
    y_num[y=='b'] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return y_num, x, indices

def impute_missing(x, x_test):
    """
    Impute missing values: Delete features with more than 80% missing values
                           Impute the mode in the features with less than 80% missing values 
    """
    num_row, num_col = x.shape # N number of rows, D number of cols
    count_missing = np.zeros(num_col) #create a vector with len equal to the number of variables
    remove = [] # list indicating which columns we should remove
    for i in range(num_col):
        count_missing[i] = np.count_nonzero(x[:,i] == -999) # compute num of missing values per column
      
        if (count_missing[i]/num_row > 0.7): 
            remove.append(i) # se maggiore di .7 allora aggiungere alla lista di colonne da rimuovere
           
        elif count_missing[i]>0: # se minore di .8 ma comunque con missing presenti: imputare missing con mediana!
            col_i = x[:,i] 
            median = np.median(col_i[col_i != -999])
            x[:,i] = np.where(x[:,i]==-999, median, x[:,i])
            x_test[:,i] = np.where(x_test[:,i]==-999, median, x_test[:,i])
                    
    x[:,remove]=0
    x_test[:,remove]=0
        
    return x, x_test

def hist_plot_jet_class(y,x):
    jet_class = {
        0: x[:, 22] == 0,
        1: x[:, 22] == 1,
        2: x[:, 22] == 2, 
        3: x[:, 22] == 3
        }

    plot = plt.subplot(111)
    colors = ['red','blue','green','orange']
    legend = ['calss: 0','class: 1','class: 2','class: 3']
    pred = np.array([-1,  1])
    w = 0.4
    for class_i in range(len(jet_class)):
        y_class_i = y[jet_class[class_i]]
        count_prediction = {-1:  np.count_nonzero(y_class_i == -1), 1:  np.count_nonzero(y_class_i == 1)}
        plot.bar(pred+w*class_i, count_prediction.values(), width=w, color=colors[class_i],align='center')

    plot.set_ylabel('count')
    plot.set_xticks(pred+w)
    plot.set_xticklabels( ('prediction is -1', 'prediction is 1') )
    plot.legend(legend)
    plot.plot()
    
def distributionsPlot(y,x,colnames):

    alphaQuantile = 0

    for i in range(len(colnames)):

        y_i =  y[(x[:,i] != - 999.0)]
        x_i = x[(x[:,i] != - 999.0),:]
        
        if x.shape[0]!=0:

            positive_ones = [y_i==1][0]
            negative_ones = [y_i==-1][0]

            plt.hist(x_i[positive_ones,i] ,100, histtype ='step',color='red',label='y == 1',density=True)      
            plt.hist(x_i[negative_ones,i] ,100, histtype ='step',color='blue',label='y == -1',density=True)  
            plt.legend(loc = "upper right")
            plt.title("{name}, feature: {id}/{tot}".format(name=colnames[i],id=i,tot=len(colnames)-1), fontsize=12)
            plt.show()        

def outliers(x, alpha=0):
    """
   if a value is smaller than the alpha_percentile we replace it with that percentile. Same if the value is larger than the 1-       alpha -percentile   """
    for i in range(x.shape[1]):
        x[:,i][ x[:,i]<np.percentile(x[:,i],alpha) ] = np.percentile(x[:,i],alpha)
        x[:,i][ x[:,i]>np.percentile(x[:,i],100-alpha) ] = np.percentile(x[:,i],100-alpha)
        
    return x

def standardize(x, mean_x=None, std_x=None):
    """ Standardize the dataset """
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x = x[:, std_x > 0] / std_x[std_x > 0]

    return x, mean_x, std_x

def compute_accuracy(y_pred, y):
    """Computes accuracy"""
    sum = 0
    for idx, y_val in enumerate(y):
        if y_val == y_pred[idx]:
            sum += 1

    return sum / len(y)

def transform_binary(weights, data):
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def get_predictions(x, best_w):
    preds = np.dot(x,best_w)
    y_te = np.where(preds < 0,-1,1)
    y_pred = np.insert(y_pred, 0, ["Id", "Prediction"], axis=0)
    return y_pred

# COMPUTING LOSS 
#*************************************************************************
def compute_error(y,tx,w):
    return y - tx.dot(w).reshape((y.shape[0],))
    
#def compute_mse(e):
 #   """Calculate the mse for vector e."""
  #  return 1/2*np.mean(e**2)

def compute_mse(y, tx, w):
    """
    Compute the Mean Square Error as defined in class.
    Takes as input the targeted y, the sample matrix X and the feature fector w.
    """
    e = y - tx@w
    mse = e.T.dot(e) /(2*len(e))
    return mse

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

           
            
# Preprocessing
#*************************************************************************


def standardize(x, mean_x=None, std_x=None):
    """ Standardize the dataset """
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x = x[:, std_x > 0] / std_x[std_x > 0]

    return x, mean_x, std_x


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.
        
    Returns:
        poly: numpy array of shape (N,d+1)
    """
    poly = np.ones((len(x),1))
    for j in range( 1, degree + 1):
        poly = np.c_[poly, np.power(x, j)]
    return poly


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
    pred = sigmoid(tx.dot(w))
    pred = np.diag(pred.T[0])
    r = np.multiply(pred, (1-pred)) * 1/len(y)
    return tx.T.dot(r).dot(tx)

def learning_by_gradient_descent(y, tx, w, gamma):
    """one step of gradient descent using logistic regression. Return the loss and the updated w"""
    loss= calculate_loss_lr(y, tx, w)
    w = w - gamma * calculate_gradient_lr(y, tx, w)
    return loss, w

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly





def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0] #qui in realtà sembra stia prendendo tutta y... però dipende con che parametri chiami la funzione
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
###############################################################################################

def logistic_regression(y, tx, w):
    """return the loss, gradient of the loss, and hessian of the loss."""
    loss = calculate_loss_lr(y,tx,w)
    grad = calculate_gradient_lr(y, tx, w)
    hess = calculate_hessian(y,tx,w)
    return (loss, grad, hess)
    
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

def learning_by_newton_method(y, tx, w, gamma):
    """Do one step of Newton's method. Return the loss and updated w."""
    loss, grad, hess = logistic_regression(y, tx, w)
    w_init = w
    w = w_init - gamma * np.linalg.solve(hess,grad)
    return loss, w

# GET PREDICTIONS
#*********************************************************************************************
def get_predictions_cv(x, best_w):
    preds = x.dot(best_w).reshape((x.shape[0],))
    y_pred = np.where(preds < .5,0,1)
    #print(y_pred[0:5])
    #y_pred = np.insert(y_pred, 0, ["Id", "Prediction"], axis=0)
    return y_pred

# CROSS VALIDATION
#*********************************************************************************************
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

 

"""def cross_validation(y, x, k_indices, k, lambda_, degree):
    #return the loss of ridge regression for a fold corresponding to k_indicec
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
    
    return loss_tr, """

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

# SCORING

