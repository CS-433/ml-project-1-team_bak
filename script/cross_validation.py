def cross_validation_jet(y, x, method, k_indices, k, degrees, alphas, lambdas=None, log=False, **kwargs):
    """
    Completes k-fold cross-validation for Least Squares with GD, SGD, Normal Equations, Logistic and Regularized Logistic 
    Regression with SGD
    """
    # get k'th subgroup in test, others in train
    test_indeces = k_indices[k] # molto semplicemente prende il gruppo k-esimo e lo mette come test
    train_indeces = np.delete(k_indices, (k), axis=0).ravel() # qua invece prende tutti gli altri gruppi e li usa come train

    x_train_all_jets = x[train_indeces, :]
    x_test_all_jets = x[test_indeces, :]
    y_train_all_jets = y[train_indeces]
    y_test_all_jets = y[test_indeces]

    # split in 4 subsets the training set accordingly to JET class
    jet_train_class = {
        0: x_train_all_jets[:, 22] == 0,
        1: x_train_all_jets[:, 22] == 1,
        2: x_train_all_jets[:, 22] == 2, 
        3: x_train_all_jets[:, 22] == 3
    }
    
    jet_test_class = {
        0: x_test_all_jets[:, 22] == 0,
        1: x_test_all_jets[:, 22] == 1,
        2: x_test_all_jets[:, 22] == 2, 
        3: x_test_all_jets[:, 22] == 3
    }


    # initialize output vectors
    y_train_pred = np.zeros(len(y_train_all_jets))
    y_test_pred = np.zeros(len(y_test_all_jets))

    for idx in range(len(jet_train_class)):
        x_train = x_train_all_jets[jet_train_class[idx]]
        x_test = x_test_all_jets[jet_test_class[idx]]
        y_train = y_train_all_jets[jet_train_class[idx]]

        # data pre-processing
        x_train, x_test = pre_process_data(x_train, x_test, alphas[idx])
        x_train = build_poly(x_train, degrees[idx]) 
        x_test = build_poly(x_test, degrees[idx]) 
        
        # compute weights using given method
        if lambdas == None:
            weights, _ = method(y_train, x_train, **kwargs)
        else:
            weights, _ = method(y_train, x_train, lambdas[idx], **kwargs)
        
        # predict
        if log == True:
            y_train_pred[jet_train_class[idx]] = predict_labels_logistic(weights, x_train)
            y_test_pred[jet_test_class[idx]] = predict_labels_logistic(weights, x_test)
        else:
            y_train_pred[jet_train_class[idx]] = predict_labels(weights, x_train)
            y_test_pred[jet_test_class[idx]] = predict_labels(weights, x_test)
        
    # compute accuracy for train and test data
    acc_train = compute_accuracy(y_train_pred, y_train_all_jets)
    acc_test = compute_accuracy(y_test_pred, y_test_all_jets)
    
    return acc_train, acc_test
