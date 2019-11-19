import pandas as pd
import os
import geopy.distance

from problem_manager import Problem
from external_data_generator import ExternalDataGenerator

class DataManager():

    def __init__(self):
        print('Initializing data manager...')
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
        print("Retrieving external_data...")
        edg = ExternalDataGenerator(read_only = True)
        self._external_data = edg.get_data()

    def suffix_join(self, X, additional, suffix, col) :
        additional.columns += suffix

        # Renaming in order to join.
        additional = additional.rename(
            columns={
                'Date' + suffix : 'DateOfDeparture', 
                'AirPort' + suffix : col
            }
        )
        
        new_X = pd.merge(
            X, additional, how='left',
            left_on=['DateOfDeparture', col],
            right_on=['DateOfDeparture', col],
            sort=False
        )

        additional.columns = additional.columns.str.replace(suffix, '')
        additional = additional.rename(
            columns={
                'DateOfDeparture' : 'Date', 
                col : 'AirPort'
            }
        )
        return new_X, additional

    def transform(self) :
        """
        Data reformatter, making the data ready for model fitting.
        """
        print("Transforming data...")
        train_X = self._train_X
        train_X['label'] = 'train'
        test_X = self._test_X
        test_X['label'] = 'test'
        
        new_X = pd.concat([train_X , test_X])

        # We keep only the columns that are relevant for our model.
        external_data = self._external_data.filter(
            self._problem.get_ed_model_columns()
        )
        
        
        new_X, external_data = self.suffix_join(new_X, external_data, '_dep', 'Departure')
        new_X, external_data = self.suffix_join(new_X, external_data, '_arr', 'Arrival')

        # Distanc between 2 airports 
        dep_coords = list(
            zip(
                new_X.loc[:, 'coordinates_dep'].apply(lambda x: float(x.split(", ")[0])),
                new_X.loc[:, 'coordinates_dep'].apply(lambda x: float(x.split(", ")[1]))
            )
        )

        arr_coords = list(
            zip(
                new_X.loc[:, 'coordinates_arr'].apply(lambda x: float(x.split(", ")[0])),
                new_X.loc[:, 'coordinates_arr'].apply(lambda x: float(x.split(", ")[1]))
            )
        )
        
        new_X.loc[:, 'distance'] = [geopy.distance.distance(dep, arr).km for dep, arr in zip(dep_coords, arr_coords)]

        new_X.drop(['DateOfDeparture', 'coordinates_dep', 'coordinates_arr'], axis = 1, inplace = True)

        # Creating dummy variables.
        to_dumify = {
            'Events_dep' : 'e_d',
            'type_dep' : 't_d',
            'Events_arr' : 'e_a',
            'type_arr' : 't_a',
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

        self._train_X = new_X[new_X['label'] == 'train'].drop('label', axis = 1)
        self._test_X = new_X[new_X['label'] == 'test'].drop('label', axis = 1)
        return