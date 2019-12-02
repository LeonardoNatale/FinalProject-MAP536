from Service.data_manager import DataManager
from Service.ramp_external_data_generator import RampExternalDataGenerator
from Service.problem_manager import Problem
import pandas as pd
import geopy


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

        new_x, external_data = DataManager.suffix_join(new_x, external_data, '_dep', 'Departure')
        new_x, external_data = DataManager.suffix_join(new_x, external_data, '_arr', 'Arrival')

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
            self.__edg.get_log_pax(),
            how='left',
            on=["Departure", "DateOfDeparture"]
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
