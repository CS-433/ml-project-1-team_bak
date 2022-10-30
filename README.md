## CS433 - Machine Learning Project 1 
by Team BAK

## Authors (team: Team_BAK)
- Elif Kurtay
- Ernesto Bocini
- Abdullah Aydemir

## File structure
- cross_validation.py
    - File containing functions for splitting data for cross validation to choose best parameters and to perform the final training to retrieve predictions.
- helpers.py
    - File that contains various helper functions for the project generally including loss, gradient, and accuracy computations.
- implementations.py
    - File containing all 6 implementations of ML functions required for the project.
- pre_process.py
    - File containing functions to load and preprocess the data.
- plot_helpers.py
    - File containing plotting functions that are used in Plots.ipynb. 
- Training.ipynb
    - File where the training set is used to find the best hyperparameters using k-fold cross-validation
- Plots.ipynb
    - File where the ploting functions are used to show information about the data and about our models' results.
- run.py
    - Main script - training the best model on the train set using the best hyperparameters and using the test set to make predictions for the submission

## How to reproduce our results
We assume that the repository is already downloaded and extracted, that the [data](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files) is downloaded and extracted in the script folder at the root of the program. We further assume that Anaconda is already installed.

### Create the environment
Make sure your environment satisfies the following requirements:
- Python 3.7+
- NumPy module 
- matplotlib

### Run the code
From the root folder of the project

```shell
python run.py
```

