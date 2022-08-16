from sklearn import datasets, linear_model, svm, neighbors, ensemble
from sklearn.metrics import mean_squared_error
import sklearn.pipeline
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split

N_SAMPLES = 1000
TEST_FRACTION = .2
OUTPUTDIRECTORY = os.getcwd() + '/'

PARAMETERS_GRIDSEARCHCV = {'cv': 5,
                           'verbose': 1
                           }

a = sklearn.svm.SVR().get_params()
print(a)