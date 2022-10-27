def select_parameters_ridge_regression_jet(y,x,degrees,lambdas,alphas,k_fold,seed):
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
        0: x[:, 22] == 0,
        1: x[:, 22] == 1,
        2: x[:, 22] == 2, 
        3: x[:, 22] == 3
        }

    for idx in range(len(jet_class)):
        x_jet = x[jet_class[idx]]
        y_jet = y[jet_class[idx]]
        
        degree,lamb,alpha,accu = select_parameters_ridge_regression(degrees, lambdas, alphas, k_fold, y_jet, x_jet, seed)
        par_degree.append(degree)
        par_lamb.append(lamb)
        par_alpha.append(alpha)
        accuracy.append(accu)

    return par_degree, par_lamb, par_alpha, accuracy

def select_parameters_ridge_regression(degrees, lambdas, alphas, k_fold, y, x, seed):
    """
    Given the training set and a set of tuples of parameters (alphas, lamdas, degrees) 
    returns the tuple which maximize the accuracy predicted through Cross Validation 
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    best_param = []

    for degree in degrees:
        for lamb in lambdas:
            for alpha in alphas:
                accuracy_test = []
                for k in range(k_fold):
                        _, acc_test = cross_validation(y, x, ridge_regression, k_indices, k, degree, alpha, lamb)
                        accuracy_test.append(acc_test)
                best_param.append([degree,lamb,alpha,np.mean(accuracy_test)])
    
    best_param = np.array(best_param)
    ind_best =  np.argmax(best_param[:,3])  #param that maximizes the accuracy    
    best_degree = best_param[ind_best,0]
    best_lamb = best_param[ind_best,1]
    best_alpha = best_param[ind_best,2]
    accu = best_param[ind_best,3]
   
    return best_degree, best_lamb, best_alpha, accu

def cross_validation(y, x, method, k_indices, k, degree, alpha, lamb=None, log=False, **kwargs):
    """k-fold cross-validation for the different methods: LS with GD, LS with SGD, Normal Equations, Logistic and Regularized Logistic Regression with SGD"""
    # get k'th subgroup in test, others in train
    test_indeces = k_indices[k] # molto semplicemente prende il gruppo k-esimo e lo mette come test
    train_indeces = np.delete(k_indices, (k), axis=0).ravel() # qua invece prende tutti gli altri gruppi e li usa come train

    x_train = x[train_indeces, :] # crea i data set
    x_test = x[test_indeces, :]
    y_train = y[train_indeces] 
    y_test = y[test_indeces] 

    # initialize output vectors
    y_train_pred = np.zeros(len(y_train)) # crea due vettori vuoti che poi conterranno le previsioni per il train e per il test
    y_test_pred = np.zeros(len(y_test))
 
    # data pre-processing
    x_train, x_test = pre_process_data(x_train, x_test, alpha) # qui fa il preprocessing 
            
    # transformation
    x_train = build_poly(x_train, degree) 
    x_test = build_poly(x_test, degree) 
        
    # compute weights using given method
    if lamb == None:
        weights, _ = method(y_train, x_train, **kwargs) 
    else: 
        weights, _ = method(y_train, x_train, lamb, **kwargs) # ridge regression in this case
       
    # predict
    if log == True: # quindi se abbiamo a che fare con logistic
        y_train_pred = predict_labels_logistic(weights, x_train) # applica la funzione predict_labels_logistic
        y_test_pred = predict_labels_logistic(weights, x_test)
        print(y_train_pred, y_train)
    else:
        y_train_pred = transform_binary(weights, x_train) # se non è una logistic regression applica predict_labels
        y_test_pred = transform_binary(weights, x_test)
        

    # compute accuracy for train and test data
    acc_train = compute_accuracy(y_train_pred, y_train) # qua applica compute_accuracy function
                                                           
    acc_test = compute_accuracy(y_test_pred, y_test)
    
    return acc_train, acc_test