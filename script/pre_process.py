import numpy as np


def load_csv_data(data):
    """Loads data and returns y (class labels), x (features) and indices (event ids)"""
    y = np.genfromtxt(data, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data, delimiter=",", skip_header=1)
    indices = x[:, 0].astype(np.int)
    x = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    y_num = np.ones(len(y))
    y_num[y=='b'] = -1
    return y_num, x, indices


def save_predictions(pred, inds_test, title="submission"):
    pred = pred.astype(int)
    y_pred = np.c_[inds_test, pred].astype(str)
    y_pred = np.insert(y_pred, 0, ["Id", "Prediction"], axis=0)
    np.savetxt("../data/" + title + ".csv", y_pred, fmt="%s", delimiter=",")
    

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
            remove.append(i)
           
        elif count_missing[i]>0: # if less than .7, impute missing values with median
            col_i = x[:,i] 
            median = np.median(col_i[col_i != -999])
            x[:,i] = np.where(x[:,i]==-999, median, x[:,i])
            x_test[:,i] = np.where(x_test[:,i]==-999, median, x_test[:,i])
                    
    x[:,remove]=0
    x_test[:,remove]=0
        
    return x, x_test


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


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, int(degree)+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def pre_process_data(x_train, x_test):
    """
    With this preprocess we are going to:
    - impute missing values with the median
    - perform feature study and remove variables that are useless
    - impute outliers using alpha quantiles
    - standardize data
    """
    # Impute missing data
    x_train, x_test = impute_missing(x_train, x_test)
    
    # remove the features that have low information gain, check plot file
    x_train = np.delete(x_train, [15,16,18,19,20,21], 1) 
    x_test = np.delete(x_test, [15,16,18,19,20,21], 1)
    
    # TODO coliearity
    return x_train, x_test


def modify_data(x_train, x_test, alpha=0, degree=2):
    """
    With this preprocess we are going to:
    - impute outliers using alpha quantiles
    - standardize data
    """
    # Delete outliers
    x_train = outliers(x_train, alpha)
    x_test = outliers(x_test, alpha)
    
    # Standardization
    x_train, mean_x_train, std_x_train = standardize(x_train) 
    x_test, _, _ = standardize(x_test, mean_x_train, std_x_train)
    
    # transformation
    x_train = build_poly(x_train, degree) 
    x_test = build_poly(x_test, degree)
    
    return x_train, x_test


def get_median(a):
    a[a == -999.0] = np.nan
    median=np.nanmedian(a)
    return median
    
    
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
        for j in range(i+1,array_of_features[0,:].size):
            num_samples = len(y)
            tx = np.c_[np.ones(num_samples), array_of_features[:,j]]
            pt_w,mse=least_squares(y,tx)
            R_square=1-mse/np.var(y)
            VIF=1/(1-R_square)
            if VIF >= critical_value:
                features_VIF=np.append(features_VIF, [(str(i+1)+"-"+str(j+1)),VIF])
                
    return features_VIF

