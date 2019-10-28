import pandas as pd
import os

from problem_manager import Problem
from external_data_generator import ExternalDataGenerator

class DataManager():

    def __init__(self):
        self._problem = Problem()
        self._train_X, self._train_y = self._read_train_data()
        self._test_X, self._test_y = self._read_test_data()
        self._set_external_data()
        self.transform()
        

    def _read_data(self, path, f_name):
        """
        Reads data from a file and returns the predictors
        as a data frame and the variable to predict as a 
        Series.
        """
        data = pd.read_csv(os.path.join(path, 'data', f_name))
        y_array = data[self._problem.get_target_column_name()].values
        X_df = data.drop(self._problem.get_target_column_name(), axis=1)
        return X_df, y_array


    def _read_train_data(self, path='.'):
        """
        Returns the training data for the model.
        """
        return self._read_data(path, self._problem.get_train_f_name())


    def _read_test_data(self, path='.'):
        """
        Returns the testing data for the model.
        """
        return self._read_data(path, self._problem.get_test_f_name())

    def _set_external_data(self) :
        edg = ExternalDataGenerator()
        self._external_data = edg.get_data()

    def transform(self) :
        """
        Data reformatter, making the data ready for model fitting.
        """
        new_X = self._train_X

        # We keep only the columns that are relevant for our model.
        external_data = self._external_data.filter(
            self._problem.get_ed_model_columns()
        )
        
        # Renaming in order to join.
        external_data = external_data.rename(
            columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'})

        new_X = pd.merge(
            new_X, external_data, how='left',
            left_on=['DateOfDeparture', 'Arrival'],
            right_on=['DateOfDeparture', 'Arrival'],
            sort=False)

        new_X.drop('DateOfDeparture', axis = 1, inplace = True)

        # Creating dummy variables.
        to_dumify = {
            'Events' : 'e',
            'type' : 't',
            'Departure' : 'd',
            'Arrival' : 'a'
        }
        
        # For every variable to dummify, we create dummies
        # and then remove the original variable form the data.
        for key in to_dumify.keys() :
            new_X = new_X.join(
                pd.get_dummies(
                    new_X[key], 
                    prefix = to_dumify[key]
                )
            )
            new_X.drop(key, axis = 1, inplace = True)

        self._train_X = new_X