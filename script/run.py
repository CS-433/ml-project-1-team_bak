import numpy as np
from cross_validation import *
from implementations import *
from helpers import *
from pre_process import *

if __name__ == '__main__':
    # Load and preprocess data
    y, x, ids = load_csv_data('train.csv')
    _, x_test, ids_test = load_csv_data('test.csv')
    x_train, x_test = pre_process_data(x, x_test)

    # CONSTANTS
    JET_COLUMN = 16

    # load best parameters
    opt_degree = [3.0, 3.0, 6.0, 6.0]
    opt_alpha = [7.0, 9.0, 5.0, 5.0]
    opt_lambda = [0.00021544346900318823, 1e-07, 4.641588833612773e-06, 0.00021544346900318823]


    jet_train_class = {
        0: x_train[:, JET_COLUMN] == 0,
        1: x_train[:, JET_COLUMN] == 1,
        2: x_train[:, JET_COLUMN] == 2, 
        3: x_train[:, JET_COLUMN] == 3
    }
    
    jet_test_class = {
        0: x_test[:, JET_COLUMN] == 0,
        1: x_test[:, JET_COLUMN] == 1,
        2: x_test[:, JET_COLUMN] == 2, 
        3: x_test[:, JET_COLUMN] == 3
    }

    method_pred = np.zeros(x_test.shape[0])
    for i in range(4):
        x_jet = x_train[jet_train_class[i]]
        x_jet_test = x_test[jet_test_class[i]]
        y_jet = y[jet_train_class[i]]

        # Pre-processing and transformation of the training set and test set
        x_jet, x_jet_test = modify_data(x_jet, x_jet_test, opt_alpha[i], opt_degree[i])

        # Train the model through Ridge Regression
        best_w, _ = method(y_jet, x_jet, opt_lambda[i])

        # Prediction
        pred = get_predictions(best_w, x_jet_test)
        method_pred[jet_test_class[i]] = pred

    save_predictions(method_pred, ids_test)