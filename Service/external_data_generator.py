import os
import pandas as pd
import numpy as np
import json
from Service.holidays_manager import HolidaysManager


class ExternalDataGenerator:
    """
    This class is used to read and generate the external data.
    """

    def __init__(
            self,
            submission='my_submission',
            submissions_dir='submissions',
            data_dir='data',
            path='.',
            read_only=False
    ):
        """
        Generates the external data or reads it from a file if read_only = True.
        This class is also used as an interface for the DataManager so that it can
        easily access external data.

        :param submission: The submission related to this external data.
        :param submissions_dir: The directory where the submissions are located.
        :param data_dir: The directory where the data is located.
        :param path: The path of the root directory of the project.
        :param read_only: If true, then the data is read and not recomputed on the fly.
        """
        self._path = path
        self._data_dir = data_dir
        self._submission = submission
        self._submissions_dir = submissions_dir

        with open('./data/json/problem_config.json', 'r') as f:
            config = json.load(f)

        # Selecting the columns that we want to keep in the end.
        self._ed_model_columns = config['external_data_model_columns']
        # Additional data file names.
        self._f_names = config['ext_data_f_names']

        # If it's read only, we just read the data
        if read_only:
            self._Data = self._read_external_data()
        # Otherwise we generate the full data.
        else:
            self._Data = None
            self._generate_external_data()

    def _generate_external_data(self):
        """
        Generates the external data from the different data sources provided.
        """
        # Reading external sources of data.
        weather = self._read_file(self._f_names['weather'])
        airports = self._read_file(self._f_names['airports'])
        gdp = self._read_file(self._f_names['gdp'])
        jet_fuel = self._read_file(self._f_names['jet_fuel'])
        jet_fuel["Date"] = pd.to_datetime(jet_fuel["Date"])

        # Adding some features to the weather data.
        weather.loc[:, 'holidate'] = HolidaysManager.to_holiday(weather.loc[:, 'Date'])
        weather.loc[:, 'is_holiday'] = weather.loc[:, 'holidate'].apply(HolidaysManager.is_holiday)
        weather.loc[:, 'is_beginning_holiday'] = weather.loc[:, 'holidate'].apply(HolidaysManager.is_beginning_holiday)
        weather.loc[:, 'is_end_holiday'] = weather.loc[:, 'holidate'].apply(HolidaysManager.is_end_holiday)

        # Get distance in days to the closest holiday
        weather["Date"] = pd.to_datetime(weather["Date"])
        weather["dumb1"] = weather["Date"][weather["is_holiday"]]
        weather["dumb2"] = weather["Date"][weather["is_holiday"]]
        weather["dumb1"] = weather["dumb1"].fillna(method="ffill").fillna(method="bfill")
        weather["dumb2"] = weather["dumb2"].fillna(method="bfill").fillna(method="ffill")
        weather["distance_to_previous"] = pd.to_numeric(np.abs(weather["dumb1"] - weather["Date"]).dt.days)
        weather["distance_to_next"] = pd.to_numeric(np.abs(weather["dumb2"] - weather["Date"]).dt.days)
        weather["holidays_distance"] = np.minimum(weather.distance_to_previous, weather.distance_to_next)
        # print(weather.head())

        # weather.drop('holidate', axis=1, inplace=True)
        weather.drop(['holidate', 'dumb1', 'dumb2', 'distance_to_previous', 'distance_to_next'], axis=1, inplace=True)

        # Stripping the iso region from its prefix to allow a join on this column.
        airports.loc[:, 'iso_region'] = airports.loc[:, 'iso_region'].apply(
            lambda x: x.replace('US-', '')
        )

        # Merging the data to create external_data
        external_data = weather.merge(
            airports,
            how='left',
            left_on='AirPort',
            right_on='iata_code'
        )

        external_data = external_data.merge(
            gdp,
            how='left',
            left_on=['iso_region', 'municipality'],
            right_on=['state', 'city']
        )

        external_data = external_data.merge(
            jet_fuel,
            how='left',
            left_on=['Date'],
            right_on=['Date']
        )

        external_data['fuel_price'] = external_data['fuel_price'].fillna(method='ffill')

        rm_cols = [col for col in external_data.columns if 'Unnamed' in col]

        self._Data = external_data.drop(rm_cols, axis=1)
        self._Data = external_data.filter(
            self._ed_model_columns
        )

    def _read_file(self, file_name):
        """
        Reads the file given as a parameter and returns the content as a DataFrame.

        :param file_name: The name of the file to read.
        :return: The content of the file as a DataFrame.
        """
        return pd.read_csv(os.path.join(self._path, self._data_dir, file_name))

    def _read_external_data(self):
        """
        Reads the external data and returns it as a DataFrame.
        :return:
        """
        return pd.read_csv(
            os.path.join(
                self._path,
                self._submissions_dir,
                self._submission,
                self._f_names['external_data']
            )
        )

    def _write_external_data(self, verbose=False):
        """
        Writes the content of the _Data attribute to the submission associated with the instance.
        :param verbose: Verbose boolean.
        """
        if verbose:
            print('saving ext data')
        self._Data.to_csv(os.path.join(
            self._path,
            self._submissions_dir,
            self._submission,
            self._f_names['external_data']
        ))
        if verbose:
            print('ext data saved')

    def get_data(self):
        return self._Data

    def __str__(self):
        return self._Data.__str__()

    def head(self, n=10):
        return self._Data.head(n)
