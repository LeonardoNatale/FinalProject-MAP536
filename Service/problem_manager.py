import os
import json


class Problem:
    _problem_path = '.'
    _problem_dir = 'data/json'
    _problem_f_name = 'problem_config.json'

    def __init__(self):
        full_path = os.path.join(
            Problem._problem_path,
            Problem._problem_dir,
            Problem._problem_f_name
        )
        with open(full_path, 'r') as f:
            config = json.load(f)
        self._problem_title = config['problem_title']
        self._target_column_name = config['target_column_name']
        self._train_f_name = config['train_f_name']
        self._test_f_name = config['test_f_name']
        self._ed_model_columns = config['external_data_model_columns']

    def get_target_column_name(self):
        return self._target_column_name

    def get_train_f_name(self):
        return self._train_f_name

    def get_test_f_name(self):
        return self._test_f_name

    def get_ed_model_columns(self):
        return self._ed_model_columns
