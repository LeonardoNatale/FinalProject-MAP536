import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import geopy.distance
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor


class Problem:
    """
    This class is in charge of managing a Problem i.e. the configuration of the model implementation.
    """
    _problem_path = '.'
    _problem_dir = 'data/json'
    _problem_f_name = 'problem_config.json'

    def __init__(self):
        self._problem_title = "Number of air passengers prediction"
        self._target_column_name = "log_PAX"
        self._train_f_name = "train.csv.bz2"
        self._test_f_name = "test.csv.bz2"
        self._ed_model_columns = [
            "Date",
            "AirPort",
            "Mean TemperatureC",
            "Mean Humidity",
            "Mean Sea Level PressurehPa",
            "Mean VisibilityKm",
            "Mean Wind SpeedKm/h",
            "CloudCover",
            "is_holiday",
            "is_beginning_holiday",
            "is_end_holiday",
            "holidays_distance",
            "type",
            "gdp",
            "coordinates",
            "fuel_price",
            "passengers"
        ]
        self._ext_data_f_names = {
            "weather": "weather_data.csv",
            "airports": "airport-codes.csv",
            "gdp": "gdp.csv",
            "external_data": "external_data.csv",
            "jet_fuel": "jetfuel.csv",
            "passengers": "passengers.csv",
            "pax_aggregations": "aggregated_PAX.csv"
        }

    def get_target_column_name(self):
        return self._target_column_name

    def get_train_f_name(self):
        return self._train_f_name

    def get_test_f_name(self):
        return self._test_f_name

    def get_ed_model_columns(self):
        return self._ed_model_columns

    def get_ext_data_f_names(self):
        return self._ext_data_f_names


class RampDataManager:

    def __init__(self):
        self.__edg = RampExternalDataGenerator()
        # TODO put the attributes of problem in DM directly?
        self.__problem = Problem()
        self.categorical_columns = ['type_dep', 'type_arr']

    @staticmethod
    def suffix_join(x, additional, suffix, col):
        """
        joins the X `DataFrame` with `additional`, adding `suffix` to each column of `additional`.
        This method is precisely used to join external_data with departure and arrival.

        :param x: The DataFrame to use in the join
        :param additional: The additional data to join with.
        :param suffix: The suffix to add to `additional` columns.
        :param col: The column to rename Airport to (to make te join work)
        :return: the new X and the additional data.
        """
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

    def append_external_data(self, new_x):
        """
        Appends the external data to the X.

        :param new_x:
        :return: The new data
        """
        # We keep only the columns that are relevant for our model.
        external_data = self.__edg.get_external_data().filter(
            self.__problem.get_ed_model_columns()
        )

        new_x, external_data = RampDataManager.suffix_join(new_x, external_data, '_dep', 'Departure')
        new_x, external_data = RampDataManager.suffix_join(new_x, external_data, '_arr', 'Arrival')

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

        new_x = new_x.join(pd.get_dummies(new_x['month'], prefix='m'))
        new_x = new_x.join(pd.get_dummies(new_x['year'], prefix='y'))
        new_x = new_x.join(pd.get_dummies(new_x['day'], prefix='d'))
        new_x = new_x.join(pd.get_dummies(new_x['weekday'], prefix='wd'))
        new_x = new_x.join(pd.get_dummies(new_x['week'], prefix='w'))

        # Need to add the number of passengers here because we don't have the info in external_data_generator.
        new_x = new_x.merge(
            self.__edg.get_passengers(),
            how='left',
            left_on=['month', 'year', 'Departure', 'Arrival'],
            right_on=['month', 'year', 'origin', 'destination']
        )

        new_x = new_x.merge(
            self.__edg.get_monthly_log_pax(),
            how='left',
            on=["Departure", "month"]
        )

        new_x = new_x.merge(
            self.__edg.get_weekday_log_pax(),
            how='left',
            on=["Departure", "weekday"]
        )

        new_x.drop(
            ['DateOfDeparture', 'coordinates_dep', 'coordinates_arr', 'origin', 'destination'],
            axis=1,
            inplace=True
        )

        to_dummify = {
            'type_dep': 't_d',
            'type_arr': 't_a',
            'Departure': 'dep',
            'Arrival': 'arr'
        }

        # Add events in problemConfig
        # 'Events_dep': 'e_d',
        # 'Events_arr': 'e_a'

        # For every variable to dummify, we create dummies
        # and then remove the original variable from the data.
        for key in to_dummify.keys():
            column = new_x[key]
            categories = list(new_x[key].dropna().unique())
            column = pd.Categorical(column, categories=categories)
            new_x.drop(key, axis=1, inplace=True)
            dummies = pd.get_dummies(
                column,
                prefix=to_dummify[key]
            )
            new_x = new_x.join(
                dummies
            )

        rm_cols = [col for col in new_x.columns if 'Unnamed' in col]
        # rm_cols += [
        #     'is_holiday_arr', 'is_beginning_holiday_arr', 'is_end_holiday_arr', 'is_holiday_arr',
        #     'is_beginning_holiday_arr', 'is_end_holiday_arr', 'gdp_arr', 'fuel_price_arr'
        # ]
        # rm_cols += ['day', 'month', 'year', 'weekday', 'week']
        # rm_cols += [col for col in new_x.columns if 'PAXMean' in col]
        new_x = new_x.drop(rm_cols, axis=1)

        return new_x


class RampExternalDataGenerator:

    DataSeparator = '#-#'

    def __init__(self):
        external_data_path = os.path.join(os.path.dirname(__file__), 'external_data.csv')
        self.__external_data = pd.read_csv(external_data_path, header=0)
        self.__passengers = pd.read_csv(
            'https://raw.githubusercontent.com/guillaume-le-fur/MAP536Data/master/passengers.csv'
        )
        self.__monthly_logPAX = pd.read_csv(
            'https://raw.githubusercontent.com/guillaume-le-fur/MAP536Data/master/aggregated_monthly_PAX.csv'
        )
        self.__weekday_logPAX = pd.read_csv(
            'https://raw.githubusercontent.com/guillaume-le-fur/MAP536Data/master/aggregated_weekday_PAX.csv'
        )

    def get_external_data(self):
        return self.__external_data

    def get_passengers(self):
        return self.__passengers

    def get_monthly_log_pax(self):
        return self.__monthly_logPAX

    def get_weekday_log_pax(self):
        return self.__weekday_logPAX


class RampModel:

    def __init__(self, sk_model, fixed_parameters=None, optimizable_parameters=None):
        self._model = sk_model
        self.model_name = self._model.__name__
        self._model_name_lower = self.model_name.lower()
        self.__dm = RampDataManager()
        self.__fixed_parameters = fixed_parameters
        self.__optimizable_parameters = optimizable_parameters
        self.__pipeline = Pipeline(
            [
                (
                    self._model_name_lower,
                    self._model(**self.__fixed_parameters)
                )
            ]
        )

    def get_pipeline(self):
        return self.__pipeline

    def fit(self, x, y):
        print('Fitting training data...')
        new_x = self.__dm.append_external_data(x)
        nas = new_x.isna().any()
        self.__pipeline.fit(X=new_x, y=y)

    def predict(self, x):
        print('Making prediction...')
        new_x = self.__dm.append_external_data(x)
        return self.__pipeline.predict(X=new_x)

    def rmse(self, x, y):
        """
        Returns the RMSE of the model on test data.

        :return: The RMSE of the model.
        """
        return np.sqrt(mean_squared_error(self.predict(x), y))

    def feature_importance(self, x):
        ordering = np.argsort(self.__pipeline[0].feature_importances_)[::-1][:20]
        importances = self.__pipeline[0].feature_importances_[ordering]
        feature_names = self.__dm.append_external_data(x).columns[ordering]
        x = np.arange(len(feature_names))
        plt.figure()
        plt.bar(x, importances)
        plt.xticks(x, feature_names, rotation=90)
        plt.show()


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = RampModel(
            sk_model=AdaBoostRegressor,
            fixed_parameters={
                "base_estimator": HistGradientBoostingRegressor(
                    l2_regularization=0.9752299302272766,
                    learning_rate=0.153187560120574
                )
            },
            optimizable_parameters={}
        )
        # self.reg = RampModel(
        #     sk_model=HistGradientBoostingRegressor,
        #     fixed_parameters={
        #         "l2_regularization": 0.9752299302272766,
        #         "learning_rate": 0.153187560120574
        #     },
        #     optimizable_parameters={}
        # )

    def fit(self, X, y):
        print("fit...")
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
