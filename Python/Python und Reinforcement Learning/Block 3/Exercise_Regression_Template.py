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


def create_dataset(samples=N_SAMPLES, test_fraction=TEST_FRACTION):
    # Input:
    #   - samples: int, which specifies the number of samples in the dataset
    #   - test_fraction: float, specifies the fraction of datapoints which are used for testing
    # Return:
    #   - x_train: array, X values for training (dim: (X, 1))
    #   - x_test: array, X values for testing (dim: (X, 1))
    #   - y_train: array, Y values for training (dim: (X, ))
    #   - y_test: array, Y values for testing (dim: (X, ))
    #   - x_predict: array, X values for prediction (dim: (X, 1))
    # Function:
    #   - use datasets.make_regression() to generate a dataset with samples of datapoints; Use the following parameters:
    #       - n_features=1
    #       - n_targets=1
    #       - noise=5
    #   - split the dataset in two parts according to test_fraction
    #   - generate X values for prediction; they should be in the same interval as the X values of the generated dataset
    #     Generate as many points for prediction as there are samples in the dataset

    ### Enter your code here ###
    x, y = datasets.make_regression(n_samples=samples, n_features=1, n_targets=1, noise=5)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_fraction)
    # x_predict = (max(x) - min(x)) * np.random.random((len(x), 1)) + min(x)
    x_predict = np.linspace(min(x), max(x), len(x))
    dataset = [x_train, x_test, y_train, y_test, x_predict]

    ### End of your code ###

    return dataset


def train_lin_reg(dataset_in, filename=(OUTPUTDIRECTORY + 'lin_reg')):
    # Input:
    #   - dataset_in: list containing x_train, x_test, y_train, y_test, x_predict
    #   - filename: string
    # Return:
    #   - none
    # Function:
    #   - create 2 dicts with hyperparameters for data transformation and the regression model (already done)
    #     ('X__Y' references to a specific parameter Y of X)
    #     (arrays or lists create a grid of possible combinations which afterwards are tested;
    #      here: 2 different feature ranges)
    #   - combine the 2 dicts to a new parameter grid called parameter_dict (already done)
    #   - implement a MinMaxScaler in preprocessor using sklearn.preprocessing
    #   - create a new LinearRegression() model
    #   - define the steps of the pipeline (already done)
    #   - create the pipeline with sklearn.pipeline.Pipeline()
    #   - optimize the model using sklearn.model_selection.GridSearchCV()
    #       - Arguments: pipe, param_grid, **PARAMETERS_GRIDSEARCHCV
    #       - GridSearchCV: GridSearch on param_grid using cv-fold Cross Validation
    #         (cv is specified in PARAMETERS_GRIDSEARCH)
    #   - fit the optimized model with the train dataset
    #   - save the fitted model in filename.joblib using the joblib library

    hyperparameters_data_transformation = {'transform_data__feature_range': [(-1, 1), (0, 1)],
                                           'transform_data__copy': [True],
                                           }

    hyperparameters_model = {}

    parameter_dict = {}
    parameter_dict.update(hyperparameters_data_transformation)
    parameter_dict.update(hyperparameters_model)

    preprocessor = sklearn.preprocessing.MinMaxScaler()
    model = linear_model.LinearRegression()

    steps = [('transform_data', preprocessor), ('built_model', model)]
    pipe = sklearn.pipeline.Pipeline(steps)

    optimized_model = sklearn.model_selection.GridSearchCV(pipe, parameter_dict, **PARAMETERS_GRIDSEARCHCV)

    ### Enter your code here ###
    optimized_model.fit(dataset_in[0], dataset_in[2])
    joblib.dump(optimized_model, filename)

    ### End of your code ###

    return


def train_svr(dataset_in, filename=(OUTPUTDIRECTORY + 'SVR')):
    # Input:
    #   - dataset_in: list containing x_train, x_test, y_train, y_test, x_predict
    #   - filename: string
    # Return:
    #   - none
    # Function:
    #   - create 2 dicts with hyperparameters for data transformation and the regression model
    #     use the same dict for data transformation as in train_lin_reg()
    #     model parameters:
    #       - vary epsilon and C: use logspace from -5 to 5 to base 2 creating 10 samples
    #       - set tol to 1e-3
    #   - combine the 2 dicts to a new parameter grid called parameter_dict
    #   - implement a MinMaxScaler in preprocessor using sklearn.preprocessing
    #   - create a new SVR() model
    #   - define the steps of the pipeline
    #   - create the pipeline with sklearn.pipeline.Pipeline()
    #   - optimize the model using sklearn.model_selection.GridSearchCV()
    #       - Arguments: pipe, param_grid, **PARAMETERS_GRIDSEARCHCV
    #       - GridSearchCV: GridSearch on param_grid using cv-fold Cross Validation
    #         (cv is specified in PARAMETERS_GRIDSEARCH)
    #   - fit the optimized model with the train dataset
    #   - save the fitted model in filename.joblib using the joblib library

    ### Enter your code here ###
    hyperparameters_data_transformation = {'transform_data__feature_range': [(-1, 1), (0, 1)],
                                           'transform_data__copy': [True],
                                           }

    hyperparameters_model = {'built_model__epsilon': np.logspace(-5.0, 5.0, num=10, base=2.0),
                             'built_model__C': np.logspace(-5.0, 5.0, num=10, base=2.0),
                             'built_model__tol': [0.001]
                             }

    parameter_dict = {}
    parameter_dict.update(hyperparameters_data_transformation)
    parameter_dict.update(hyperparameters_model)

    preprocessor = sklearn.preprocessing.MinMaxScaler()
    model = sklearn.svm.SVR()

    steps = [('transform_data', preprocessor), ('built_model', model)]
    pipe = sklearn.pipeline.Pipeline(steps)

    optimized_model = sklearn.model_selection.GridSearchCV(pipe, parameter_dict, **PARAMETERS_GRIDSEARCHCV)

    optimized_model.fit(dataset_in[0], dataset_in[2])
    joblib.dump(optimized_model, filename)

    ### End of your code ###

    return


def train_random_forest(dataset_in, filename=(OUTPUTDIRECTORY + 'Rand_for')):
    # Input:
    #   - dataset_in: list containing x_train, x_test, y_train, y_test, x_predict
    #   - filename: string
    # Return:
    #   - none
    # Function:
    #   - create 2 dicts with hyperparameters for data transformation and the regression model
    #     use the same dict for data transformation as in train_lin_reg()
    #     model parameters:
    #       - vary max_features between 'auto' and 'sqrt'
    #       - set n_estimators to 5000
    #   - combine the 2 dicts to a new parameter grid called parameter_dict
    #   - implement a MinMaxScaler in preprocessor using sklearn.preprocessing
    #   - create a new RandomForestRegressor() model
    #   - define the steps of the pipeline
    #   - create the pipeline with sklearn.pipeline.Pipeline()
    #   - optimize the model using sklearn.model_selection.GridSearchCV()
    #       - Arguments: pipe, param_grid, **PARAMETERS_GRIDSEARCHCV
    #       - GridSearchCV: GridSearch on param_grid using cv-fold Cross Validation
    #         (cv is specified in PARAMETERS_GRIDSEARCH)
    #   - fit the optimized model with the train dataset
    #   - save the fitted model in filename.joblib using the joblib library

    ### Enter your code here ###
    hyperparameters_data_transformation = {'transform_data__feature_range': [(-1, 1), (0, 1)],
                                           'transform_data__copy': [True],
                                           }

    hyperparameters_model = {'built_model__max_features': ['auto', 'sqrt'],
                             'built_model__n_estimators': [5000]}

    parameter_dict = {}
    parameter_dict.update(hyperparameters_data_transformation)
    parameter_dict.update(hyperparameters_model)

    preprocessor = sklearn.preprocessing.MinMaxScaler()
    model = sklearn.ensemble.RandomForestRegressor()

    steps = [('transform_data', preprocessor), ('built_model', model)]
    pipe = sklearn.pipeline.Pipeline(steps)

    optimized_model = sklearn.model_selection.GridSearchCV(pipe, parameter_dict, **PARAMETERS_GRIDSEARCHCV)

    optimized_model.fit(dataset_in[0], dataset_in[2])
    joblib.dump(optimized_model, filename)

    ### End of your code ###

    return


def train_knn(dataset_in, filename=(OUTPUTDIRECTORY + 'KNN')):
    # Input:
    #   - dataset_in: list containing x_train, x_test, y_train, y_test, x_predict
    #   - filename: string
    # Return:
    #   - none
    # Function:
    #   - create 2 dicts with hyperparameters for data transformation and the regression model (already done)
    #     use the same dict for data transformation as in train_lin_reg()
    #     model parameters:
    #       - have a look at the documentation of KNeighborsRegression() and select 2 parameters to vary
    #   - combine the 2 dicts to a new parameter grid called parameter_dict (already done)
    #   - implement a MinMaxScaler in preprocessor using sklearn.preprocessing
    #   - create a new KNeighborsRegression() model
    #   - define the steps of the pipeline
    #   - create the pipeline with sklearn.pipeline.Pipeline()
    #   - optimize the model using sklearn.model_selection.GridSearchCV()
    #       - Arguments: pipe, param_grid, **PARAMETERS_GRIDSEARCHCV
    #       - GridSearchCV: GridSearch on param_grid using cv-fold Cross Validation
    #         (cv is specified in PARAMETERS_GRIDSEARCH)
    #   - fit the optimized model with the train dataset
    #   - save the fitted model in filename.joblib using the joblib library

    ### Enter your code here ###
    hyperparameters_data_transformation = {'transform_data__feature_range': [(-1, 1), (0, 1)],
                                           'transform_data__copy': [True],
                                           }

    hyperparameters_model = {'built_model__n_neighbors': [5],
                             'built_model__p': [2]}

    parameter_dict = {}
    parameter_dict.update(hyperparameters_data_transformation)
    parameter_dict.update(hyperparameters_model)

    preprocessor = sklearn.preprocessing.MinMaxScaler()
    model = neighbors.KNeighborsRegressor()

    steps = [('transform_data', preprocessor), ('built_model', model)]
    pipe = sklearn.pipeline.Pipeline(steps)

    optimized_model = sklearn.model_selection.GridSearchCV(pipe, parameter_dict, **PARAMETERS_GRIDSEARCHCV)

    optimized_model.fit(dataset_in[0], dataset_in[2])
    joblib.dump(optimized_model, filename)

    ### End of your code ###

    return


def model_predict(dataset_in, filename_extension):
    # Input
    #   - dataset_in: list containing x_train, x_test, y_train, y_test, x_predict
    #   - filename_extension: string, specifing which file to open
    # Return:
    #   - mse: float, mean squared error
    #   - y: array, predicted y values based on x_predict
    # Function:
    #   - load model from file
    #   - compute and return mse evaluating the prediction for x_test and y_test
    #   - compute and return the prediction for x_predict

    ### Enter your code here ###
    model_sel = joblib.load(filename_extension)
    y_pred = model_sel.predict(dataset_in[1])
    mse = mean_squared_error(dataset_in[3], y_pred)

    y = model_sel.predict(dataset_in[4])

    ### End of your code ###
    
    return mse, y


if __name__ == "__main__":
    # Create dataset
    dataset = create_dataset()

    # plot dataset
    plt.plot(dataset[0], dataset[2], '.', color='blue', ms=2)
    plt.plot(dataset[1], dataset[3], '.', color='blue', ms=2, label='Database')

    # train different models
    train_lin_reg(dataset)
    train_svr(dataset)
    train_random_forest(dataset)
    train_knn(dataset)

    models = [('lin_reg', 'Linear Regression'), ('SVR', 'SVR'), ('rand_for', 'Random Forest'), ('KNN', 'KNN')]
    results = []

    print('MSE for different models:')
    for model_sel in models:
        results.append(model_predict(dataset, model_sel[0]))
        plt.plot(dataset[4], results[-1][1], label=model_sel[1])
        print('MSE for {0} = {1:.2f}'.format(model_sel[1], results[-1][0]))

    plt.legend(loc='upper left')
    plt.show()
