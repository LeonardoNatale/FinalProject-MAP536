import os
import pandas as pd
import numpy as np
import rampwf as rw
import holidays
import datetime
from sklearn.model_selection import ShuffleSplit
from service.holidays_manager import HolidaysManager
from service.external_data_generator import ExternalDataGenerator

problem_title = 'Number of air passengers prediction'
_target_column_name = 'log_PAX'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow
workflow = rw.workflows.AirPassengers()

score_types = [
    rw.score_types.RMSE(name='rmse', precision=3),
]

def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.5, random_state=57)
    return cv.split(X)


def _read_data(path, f_name):
    """
    Reads data from a file and returns the predictors
    as a data frame and the variable to predict as a 
    Series.
    """
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    """
    Returns the training data for the model.
    """
    f_name = 'train.csv.bz2'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    """
    Returns the testing data for the model.
    """
    f_name = 'test.csv.bz2'
    return _read_data(path, f_name)


def generate_external_data(path = '.'):
    """
    This function returns the external data that 
    is going to be used for the model.
    """
    weather_f_name = 'weather_data.csv'
    airports_f_name = 'airport-codes.csv'
    gdp_f_name = 'gdp.csv'
    external_data_f_name = 'external_data.csv'

    # TODO make the external data generator dependent on the submission
    # by creating a config file containing all the infos about the submission.
    dg = ExternalDataGenerator()
    dg._generate_external_data()
    print(dg.head())

generate_external_data()