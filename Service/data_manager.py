import pandas as pd
import os
import geopy.distance

from Service.problem_manager import Problem
from Service.external_data_generator import ExternalDataGenerator

class DataManager:

    def __init__(self, ramp = False):
        self.__ramp = ramp
        self._problem = Problem()
        self._set_external_data()
        print('Initializing data manager...')
        train_x, self.__train_y = self._read_train_data()
        train_x['label'] = 'train'
        test_x, self.__test_y = self._read_test_data()
        test_x['label'] = 'test'
        self.__full_X = pd.concat([train_x, test_x])
        self.__train_X = self.__full_X[self.__full_X['label'] == 'train'].drop('label', axis=1)
        self.__test_X = self.__full_X[self.__full_X['label'] == 'test'].drop('label', axis=1)

        self.transform()

    def get_test_X(self):
        return self.__test_X

    def get_test_y(self):
        return self.__test_y

    def get_train_X(self):
        return self.__train_X

    def get_train_y(self):
        return self.__train_y

    def _read_data(self, path, f_name):
        """
        Reads data from a file and returns the predictors
        as a data frame and the variable to predict as a 
        Series.
        """
        data = pd.read_csv(os.path.join(path, 'data', f_name))
        y_array = data[self._problem.get_target_column_name()].values
        x_df = data.drop(self._problem.get_target_column_name(), axis=1)
        return x_df, y_array

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

    def _set_external_data(self):
        print("Retrieving external_data...")
        edg = ExternalDataGenerator(read_only=True)
        self._external_data = edg.get_data()

    def suffix_join(self, x, additional, suffix, col):
        additional.columns += suffix

        # Renaming in order to join.
        additional = additional.rename(
            columns={
                'Date' + suffix: 'DateOfDeparture',
                'AirPort' + suffix: col
            }
        )

        new_x = pd.merge(
            x, additional, how='left',
            left_on=['DateOfDeparture', col],
            right_on=['DateOfDeparture', col],
            sort=False
        )

        additional.columns = additional.columns.str.replace(suffix, '')
        additional = additional.rename(
            columns={
                'DateOfDeparture': 'Date',
                col: 'AirPort'
            }
        )
        return new_x, additional

    def append_to_data(self, data, full = False):
        new_x = data

        # We keep only the columns that are relevant for our model.
        external_data = self._external_data.filter(
            self._problem.get_ed_model_columns()
        )

        new_x, external_data = self.suffix_join(new_x, external_data, '_dep', 'Departure')
        new_x, external_data = self.suffix_join(new_x, external_data, '_arr', 'Arrival')

        # Distance between 2 airports
        dep_coords = list(
            zip(
                new_x.loc[:, 'coordinates_dep'].apply(lambda x: float(x.split(", ")[0])),
                new_x.loc[:, 'coordinates_dep'].apply(lambda x: float(x.split(", ")[1]))
            )
        )

        arr_coords = list(
            zip(
                new_x.loc[:, 'coordinates_arr'].apply(lambda x: float(x.split(", ")[0])),
                new_x.loc[:, 'coordinates_arr'].apply(lambda x: float(x.split(", ")[1]))
            )
        )

        new_x.loc[:, 'distance'] = [geopy.distance.distance(dep, arr).km for dep, arr in zip(dep_coords, arr_coords)]

        new_x['DateOfDeparture'] = pd.to_datetime(new_x['DateOfDeparture'])
        new_x['year'] = new_x['DateOfDeparture'].dt.year
        new_x['month'] = new_x['DateOfDeparture'].dt.month
        new_x['day'] = new_x['DateOfDeparture'].dt.day
        new_x['weekday'] = new_x['DateOfDeparture'].dt.weekday
        new_x['week'] = new_x['DateOfDeparture'].dt.week
        new_x['n_days'] = new_x['DateOfDeparture'].apply(
            lambda date: (date - pd.to_datetime("1970-01-01")).days)

        new_x = new_x.join(pd.get_dummies(new_x['year'], prefix='y'))
        new_x = new_x.join(pd.get_dummies(new_x['month'], prefix='m'))
        new_x = new_x.join(pd.get_dummies(new_x['day'], prefix='d'))
        new_x = new_x.join(pd.get_dummies(new_x['weekday'], prefix='wd'))
        new_x = new_x.join(pd.get_dummies(new_x['week'], prefix='w'))

        # Need to add the number of passengers here because we don't have the info in external_data_generator.
        passengers = pd.read_csv(os.path.join('./data', self._problem.get_ext_data_f_names()['passengers']))
        new_x = new_x.merge(
            passengers,
            how='left',
            left_on=['month', 'year', 'Departure', 'Arrival'],
            right_on=['month', 'year', 'origin', 'destination']
        )

        new_x.drop(['DateOfDeparture', 'coordinates_dep', 'coordinates_arr', 'origin', 'destination'], axis=1, inplace=True)

        # Creating dummy variables.
        to_dummify = {
            'Events_dep': 'e_d',
            'type_dep': 't_d',
            'Events_arr': 'e_a',
            'type_arr': 't_a',
            'Departure': 'dep',
            'Arrival': 'arr'
        }

        # For every variable to dummify, we create dummies
        # and then remove the original variable form the data.
        for key in to_dummify.keys():
            column = new_x[key]
            if not full and self.__ramp:
                categories = list(self.__full_X[key].dropna().unique())
                column = pd.Categorical(column, categories=categories)
                new_x.drop(key, axis=1, inplace=True)
            if not self.__ramp:
                new_x.drop(key, axis=1, inplace=True)
            dummies = pd.get_dummies(
                column,
                prefix=to_dummify[key]
            )
            new_x = new_x.join(
                dummies
            )
        return new_x

    def transform(self):
        """
        Data reformatter, making the data ready for model fitting.
        """
        print("Transforming data...")
        self.__full_X = self.append_to_data(self.__full_X, full=True)
        self.__train_X = self.__full_X[self.__full_X['label'] == 'train'].drop('label', axis=1)
        self.__test_X = self.__full_X[self.__full_X['label'] == 'test'].drop('label', axis=1)
        return
