from Service.external_data_generator import ExternalDataGenerator
import os
import pandas as pd


class RampExternalDataGenerator:

    DataSeparator = '#-#'

    def __init__(
            self,
            submission='my_submission',
            submissions_dir='submissions',
            path='.'
    ):
        self.external_data_path = os.path.join(path, submissions_dir, submission, 'external_data.csv')
        self.__external_data = pd.read_csv(self.external_data_path, header=0)
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

    def _write_external_data(self, verbose=False):
        """
        Writes the content of the _Data attribute to the submission associated with the instance.
        :param verbose: Verbose boolean.
        """
        if verbose:
            print('saving ext data')
        self.__external_data.to_csv(self.external_data_path)
        if verbose:
            print('ext data saved')
