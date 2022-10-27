def pre_process_data(x_train, x_test, alpha=0):
    """
    Preprocessing: 
    - impute missing values using median,
    - feature study: see "plots" file,
    - impute outliers using alpha-percentiles,
    - standardization
    """
    # Missing Values: 
    
    # Consider the 0s in the 'PRI_jet_all_pt' as missing values
    x_train[:,-1]=np.where(x_train[:,-1]==0, -999, x_train[:,-1])
    
    # Impute missing data
    x_train, x_test = impute_missing(x_train, x_test) # see the impute_missing function in helpfulfun
    
    # Feature study:
    # Delete useless features
    x_train = np.delete(x_train, [15,16,18,19,20,21], 1) # rimuove features inutili, guardare file plot per capire
    x_test = np.delete(x_test, [15,16,18,19,20,21], 1)
    
    # Impute outliers
    x_train = outliers(x_train, alpha) # rimuove gli outliers giudicando i percentili
    x_test = outliers(x_test, alpha)
    
    # Standardization
    x_train, mean_x_train, std_x_train = standardize(x_train) # standardizza i dati.. come abiam fatto noi
    x_test, _, _ = standardize(x_test, mean_x_train, std_x_train)
     
    return x_train, x_test
