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
        external_data_path = os.path.join(path, submissions_dir, submission, 'external_data.csv')
        self.__external_data = pd.read_csv(external_data_path, header=0)
        self.__passengers = pd.read_csv(
            'https://raw.githubusercontent.com/guillaume-le-fur/MAP536Data/master/passengers.csv'
        )
        self.__logPAX = pd.read_csv(
            'https://raw.githubusercontent.com/guillaume-le-fur/MAP536Data/master/aggregated_PAX.csv'
        )
        self.__logPAX['DateOfDeparture'] = pd.to_datetime(self.__logPAX['DateOfDeparture'])

    def get_external_data(self):
        return self.__external_data

    def get_passengers(self):
        return self.__passengers

    def get_log_pax(self):
        return self.__logPAX
