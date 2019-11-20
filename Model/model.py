import os
import json
from sklearn.pipeline import Pipeline
from Service.model_optimizer import ModelOptimizer
from Service.data_manager import DataManager

from sklearn.exceptions import NotFittedError


class Model:
    Optimizable_parameters_path = '.'
    Optimizable_parameters_dir = 'data/json'
    Model_save_dir = 'data/models'
    Optimizable_parameters_f_name = 'optimizable_parameters.json'

    def __init__(self, sk_model, dm=DataManager(), fixed_parameters={}, optimizable_parameters={}):
        self._model = sk_model
        self.model_name = self._model.__name__
        self._model_name_lower = self.model_name.lower()
        self.is_optimized = False
        self._optimal_params = {}
        self.dm = dm
        self._pipeline = None
        f = os.path.join(
            Model.Optimizable_parameters_path,
            Model.Optimizable_parameters_dir,
            Model.Optimizable_parameters_f_name
        )
        if 'RandomSearch' in optimizable_parameters.keys():
            self._random_opt_params = optimizable_parameters['RandomSearch']
        else:
            self._random_opt_params = {}
        if 'GridSearch' in optimizable_parameters.keys():
            self._grid_opt_params = optimizable_parameters['GridSearch']
        else:
            self._grid_opt_params = {}
        self._fixed_parameters = fixed_parameters
        self.build_pipeline()

    def get_pipeline(self):
        return self._pipeline

    def load_from_file(self):
        with open(os.path.join(
                Model.Optimizable_parameters_path,
                Model.Model_save_dir,
                self._model_name_lower + '.model'
        ), 'r') as f:
            params = json.load(f)

        self._fixed_parameters = params['fixed']
        self.build_pipeline()
        self._optimal_params = params['opt']
        self.update_model()
        self.is_optimized = True

    def get_fixed_parameters(self):
        return self._fixed_parameters

    def build_pipeline(self):
        self._pipeline = Pipeline(
            [
                (
                    self._model_name_lower,
                    self._model(**self._fixed_parameters)
                )
            ]
        )

    def get_optimal_parameters(self):
        if self.is_optimized:
            return self._optimal_params
        else:
            raise ValueError(
                'Model has not been optimized, could not find optimal values.')

    def get_optimal_model(self):
        if self.is_optimized:
            return self._pipeline
        else:
            raise ValueError('Model has not been optimized.')

    def update_model(self):
        self._pipeline.set_params(**self._optimal_params)

    def optimize_model(self, force=False):
        if self.is_optimized and not force:
            print(
                f'Model {self.model_name} is already optimized, doing nothing, use force = True to force reoptimization.')
        else:
            print(f'Optimizing model : {self.model_name}...')
            if self._pipeline is not None:
                mo = ModelOptimizer(self)
                # Checking that the dict is not empty
                if self._random_opt_params:
                    self._optimal_params.update(
                        mo.random_search_optimize(
                            self.dm.get_train_X(),
                            self.dm.get_train_y(),
                            self._random_opt_params
                        )
                    )
                    self.update_model()
                # Checking that the dict is not empty
                if self._grid_opt_params:
                    self._optimal_params.update(
                        mo.grid_search_optimize(
                            self.dm.get_train_X(),
                            self.dm.get_train_y(),
                            self._grid_opt_params
                        )
                    )
                    self.update_model()
                self.is_optimized = True
                print('Optimization finished.')
            else:
                raise ValueError('No pipeline is set up, nothing to optimize.')

    def save_model(self):
        model_params = {
            'fixed': self._fixed_parameters,
            'opt': self._optimal_params
        }

        with open(os.path.join(
                Model.Optimizable_parameters_path,
                Model.Model_save_dir,
                self._model_name_lower + '.model'
        ), 'w') as f:
            json.dump(model_params, f)

    def fit(self):
        if not self.is_optimized:
            print(f'Running a fit on a non optimized model : {self.model_name}')
        print(f'Fitting training data for model {self.model_name}...')
        self._pipeline.fit(X=self.dm.get_train_X(), y=self.dm.get_train_y())

    def r2_score(self):
        test_score = self._pipeline.score(X=self.dm.get_test_X(), y=self.dm.get_test_y())
        print(f'R^2 coefficient : {test_score}')

    def predict(self):
        try:
            pred = self._pipeline.predict(self.dm.get_test_X())
        except NotFittedError as e:
            print(f'Model {self.model_name} not fitted. {e}')
            return None
        return pred
