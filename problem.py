import os
import pandas as pd
import numpy as np
import rampwf as rw
import holidays
import datetime
from sklearn.model_selection import ShuffleSplit

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
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv.bz2'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv.bz2'
    return _read_data(path, f_name)


def is_holiday(day):
    return day in holidays.US()


def is_beginning_holiday(day):
    day = datetime.datetime.strptime(day, '%Y-%m-%d')
    return is_holiday(day) and not is_holiday(day - datetime.timedelta(days=1))


def is_end_holiday(day):
    day = datetime.datetime.strptime(day, '%Y-%m-%d')
    return is_holiday(day) and not is_holiday(day + datetime.timedelta(days=1))

def stack_cities(row):
    if('-' in row['cities']):
        new_df = pd.DataFrame(columns=["avg", "cities","state",])
        cities = row['cities'].split('-')
        
        prt = cities
        prt.append(row['state'])
        print(prt)
        for city in cities:
            new_df.append(pd.Series([city, row['state'], row['avg']]), ignore_index=True)
        return new_df
    else :
        return row


def generate_external_data(path = '.'):
    weather_f_name = 'weather_data.csv'
    airports_f_name = 'airport-codes.csv'
    gdp_f_name = 'gdp.csv'
    weather = pd.read_csv(os.path.join(path, 'data', weather_f_name))
    airports = pd.read_csv(os.path.join(path, 'data', airports_f_name))
    gdp = pd.read_csv(os.path.join(path, 'data', gdp_f_name))

    weather.loc[:, 'is_holiday'] = weather.loc[:, 'Date'].apply(is_holiday)
    weather.loc[:, 'is_beginning_holiday'] = weather.loc[:, 'Date'].apply(is_beginning_holiday)
    weather.loc[:, 'is_end_holiday'] = weather.loc[:, 'Date'].apply(is_end_holiday) 

    external_data = weather.merge(airports, how = 'left', left_on = 'AirPort', right_on = 'iata_code')
    external_data.loc[:, 'iso_region'] = external_data.loc[:, 'iso_region'].apply(lambda x: x.replace('US-', ''))
    external_data = external_data.merge(
        gdp, 
        how = 'left', 
        left_on = ['iso_region', 'municipality'], 
        right_on = ['state', 'city']
    )

    print(external_data.filter(['Date', 'Airport', 'municipality', 'iso_region', 'gdp', 'city', 'state']).head(10))

generate_external_data()