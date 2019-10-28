import os
import pandas as pd
import holidays
import datetime
from holidays_manager import HolidaysManager

class ExternalDataGenerator():

    def __init__(
        self,
        submission = 'my_submission', 
        submissions_dir = 'submissions',
        data_dir = 'data', 
        path = '.', 
        f_names = {
            'weather' : 'weather_data.csv', 
            'airports' : 'airport-codes.csv',
            'gdp' : 'gdp.csv',
            'external_data' : 'external_data.csv'
        }
    ) :
        self._path = path
        self._data_dir = data_dir
        self._submission = submission
        self._submissions_dir = submissions_dir
        self._Data = None
        self._f_names = f_names

        # Should we call the generation of the data right away?
        self._generate_external_data()
        # What about the writing to the disk
        # self._write_external_data(self._f_names['external_data'])

    def _generate_external_data(self) :
        # Reading external sources of data.
        weather = self._read_file(self._f_names['weather'])
        airports = self._read_file(self._f_names['airports'])
        gdp = self._read_file(self._f_names['gdp'])

        # Adding some features to the weather data.
        weather.loc[:, 'holidate'] = HolidaysManager.to_holiday(weather.loc[:, 'Date'])
        weather.loc[:, 'is_holiday'] = weather.loc[:, 'holidate'].apply(HolidaysManager._is_holiday)
        weather.loc[:, 'is_beginning_holiday'] = weather.loc[:, 'holidate'].apply(HolidaysManager._is_beginning_holiday)
        weather.loc[:, 'is_end_holiday'] = weather.loc[:, 'holidate'].apply(HolidaysManager._is_end_holiday) 

        weather.drop('holidate', axis = 1, inplace = True)

        # Stripping the iso region from its prefix to allow a join on this column.
        airports.loc[:, 'iso_region'] = airports.loc[:, 'iso_region'].apply(
            lambda x: x.replace('US-', '')
        )

        # Merging the data to create external_data
        external_data = weather.merge(
            airports, 
            how = 'left', 
            left_on = 'AirPort',
            right_on = 'iata_code'
        )
        external_data = external_data.merge(
            gdp, 
            how = 'left', 
            left_on = ['iso_region', 'municipality'], 
            right_on = ['state', 'city']
        )

        rm_cols = [col for col in external_data.columns if 'Unnamed' in col]

        self._Data = external_data.drop(rm_cols, axis = 1)


    def _read_file(self, file_name):
        return pd.read_csv(os.path.join(self._path, self._data_dir, file_name))

    def _write_external_data(self, file_name = 'external_data.csv'):
        self._Data.to_csv(os.path.join(
            self._path, 
            self._submissions_dir, 
            self._submission, file_name
        ))

    def get_data(self) :
        return self._Data

    def __str__(self):
        return self._Data.__str__()

    def head(self, n = 10):
        return self._Data.head(n)

# x = ExternalDataGenerator()
# x._write_external_data()