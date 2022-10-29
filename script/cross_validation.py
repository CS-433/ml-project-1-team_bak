import numpy as np
from helpers import compute_accuracy, get_predictions, build_k_indices
from pre_process import modify_data
from implementations import *

#########################################################################
#Cross Validation#
#########################################################################
JET_COLUMN = 16

def select_parameters_jet(y,x,method,degrees,alphas,k_fold,seed, lambdas=None, log=False):
    """
    Given the training set and a set of tuples of parameters (alphas, lamdas, degrees) 
    for each jet_subset returns the tuple which maximize the accuracy predicted through Cross Validation 
    """
    par_degree = []
    par_lamb = []
    par_alpha = []
    accuracy = []

    # Split the training set in subsets according to the jet value 
    jet_class = {
        0: x[:, JET_COLUMN] == 0,
        1: x[:, JET_COLUMN] == 1,
        2: x[:, JET_COLUMN] == 2, 
        3: x[:, JET_COLUMN] == 3
        }

    for idx in range(len(jet_class)):
        x_jet = x[jet_class[idx]]
        y_jet = y[jet_class[idx]]
        
        degree,lamb,alpha,accu = select_parameters(method, degrees, lambdas, alphas, k_fold, y_jet, x_jet, seed, log)
        par_degree.append(degree)
        par_lamb.append(lamb)
        par_alpha.append(alpha)
        accuracy.append(accu)

    return par_degree, par_lamb, par_alpha, accuracy

def select_parameters(method, degrees, lambdas, alphas, k_fold, y, x, seed, log=False):
    """
    Given the training set and a set of tuples of parameters (alphas, lamdas, degrees) 
    returns the tuple which maximize the accuracy predicted through Cross Validation 
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    best_param = []
    
    if lambdas==None:
        for degree in degrees:
            for alpha in alphas:
                accuracy_test = []
                for k in range(k_fold):
                    _, acc_test = cross_validation(y, x, method, k_indices, k, degree, alpha, log=log)
                    accuracy_test.append(acc_test)
                best_param.append([degree,-1,alpha,np.mean(accuracy_test)])
    else:
        for degree in degrees:
            for lamb in lambdas:
                for alpha in alphas:
                    accuracy_test = []
                    for k in range(k_fold):
                            _, acc_test = cross_validation(y, x, method, k_indices, k, degree, alpha, lamb, log=log)
                            accuracy_test.append(acc_test)
                    best_param.append([degree,lamb,alpha,np.mean(accuracy_test)])
    
    best_param = np.array(best_param)
    ind_best =  np.argmax(best_param[:,3])  #param that maximizes the accuracy    
    best_degree = best_param[ind_best,0]
    best_lamb = best_param[ind_best,1]
    best_alpha = best_param[ind_best,2]
    accu = best_param[ind_best,3]
   
    return best_degree, best_lamb, best_alpha, accu

def cross_validation(y, x, method, k_indices, k, degree, alpha, lamb=None, log=False):
    """k-fold cross-validation for the different methods: LS with GD, LS with SGD, Normal Equations, Logistic and Regularized Logistic Regression with SGD"""
    # get k'th subgroup in test, others in train
    test_indeces = k_indices[k] 
    train_indeces = np.delete(k_indices, (k), axis=0).ravel()

    x_train = x[train_indeces, :]
    x_test = x[test_indeces, :]
    y_train = y[train_indeces] 
    y_test = y[test_indeces] 

    # initialize output vectors
    y_train_pred = np.zeros(len(y_train)) 
    y_test_pred = np.zeros(len(y_test))
 
    # data pre-processing
    x_train, x_test = modify_data(x_train, x_test, alpha, degree)
        
    # compute weights using given method
    if lamb == None:
        weights, _ = method(y_train, x_train) 
    else:  # ridge and regularized log
        weights, _ = method(y_train, x_train, lamb) 
    
    # predict
    y_train_pred = get_predictions(weights, x_train, log)
    y_test_pred = get_predictions(weights, x_test, log)
    
    # compute accuracy for train and test data
    acc_train = compute_accuracy(y_train_pred, y_train)                               
    acc_test = compute_accuracy(y_test_pred, y_test)
    
    return acc_train, acc_test


def cross_validation_result(y, x, method, k_indices, k, degrees, alphas, lambdas=None, log=False):
    """
    Completes k-fold cross-validation for Least Squares with GD, SGD, Normal Equations, Logistic and Regularized Logistic 
    Regression with SGD
    """
    # get k'th subgroup in test, others in train
    test_indeces = k_indices[k] 
    train_indeces = np.delete(k_indices, (k), axis=0).ravel() 

    x_train_all_jets = x[train_indeces, :]
    x_test_all_jets = x[test_indeces, :]
    y_train_all_jets = y[train_indeces]
    y_test_all_jets = y[test_indeces]

    # split in 4 subsets the training set accordingly to JET class
    jet_train_class = {
        0: x_train_all_jets[:, JET_COLUMN] == 0,
        1: x_train_all_jets[:, JET_COLUMN] == 1,
        2: x_train_all_jets[:, JET_COLUMN] == 2, 
        3: x_train_all_jets[:, JET_COLUMN] == 3
    }
    
    jet_test_class = {
        0: x_test_all_jets[:, JET_COLUMN] == 0,
        1: x_test_all_jets[:, JET_COLUMN] == 1,
        2: x_test_all_jets[:, JET_COLUMN] == 2, 
        3: x_test_all_jets[:, JET_COLUMN] == 3
    }


    # initialize output vectors
    y_train_pred = np.zeros(len(y_train_all_jets))
    y_test_pred = np.zeros(len(y_test_all_jets))

    for idx in range(len(jet_train_class)):
        x_train = x_train_all_jets[jet_train_class[idx]]
        x_test = x_test_all_jets[jet_test_class[idx]]
        y_train = y_train_all_jets[jet_train_class[idx]]

        # data pre-processing
        x_train, x_test = modify_data(x_train, x_test, alphas[idx], degrees[idx])
        
        # compute weights using given method
        if lambdas == None:
            weights, _ = method(y_train, x_train)
        else:
            weights, _ = method(y_train, x_train, lambdas[idx])
        
        # predict
        y_train_pred[jet_train_class[idx]] = get_predictions(weights, x_train, log)
        y_test_pred[jet_test_class[idx]] = get_predictions(weights, x_test, log)
        
    # compute accuracy for train and test data
    acc_train = compute_accuracy(y_train_pred, y_train_all_jets)
    acc_test = compute_accuracy(y_test_pred, y_test_all_jets)
    
    return acc_train, acc_test
