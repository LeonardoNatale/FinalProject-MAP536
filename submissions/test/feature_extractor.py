import pandas as pd
import os


class FeatureExtractor(object):
    def __init__(self):
        print(os.path.abspath('.'))
        current_dir_files = os.listdir("./")
        parent_dir_files = os.listdir('./../')
        ramp_data_dir = os.listdir('./ramp-data/air_passengers/')
        error_string = ''
        error_string += 'Parent dir content :\n' + str(parent_dir_files)
        error_string += '\nCurrent dir content :\n' + str(current_dir_files)
        error_string += '\nRamp dir content :\n' + str(ramp_data_dir)
        error_string += '\nFile content :\n'
        for file in current_dir_files:
            if '.py' in file or '.yml' in file:
                with(open('./' + file)) as f:
                    error_string += str(f.read())
                error_string += '\n\n-----------------\n\n'
        raise ValueError(error_string)
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df
        path = os.path.dirname(__file__)
        data_weather = pd.read_csv(os.path.join(path, 'external_data.csv'))
        X_weather = data_weather[['Date', 'AirPort', 'Mean TemperatureC']]
        X_weather = X_weather.rename(
            columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'})
        X_encoded = pd.merge(
            X_encoded, X_weather, how='left',
            left_on=['DateOfDeparture', 'Arrival'],
            right_on=['DateOfDeparture', 'Arrival'],
            sort=False)

        X_encoded = X_encoded.join(pd.get_dummies(
            X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(
            pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)

        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)
        X_array = X_encoded.values
        return X_array


