import os
import json
from sklearn.pipeline import Pipeline
from service.model_optimizer import ModelOptimizer
from service.data_manager import DataManager


class Model():

    Optimizable_parameters_path = '.'
    Optimizable_parameters_dir = 'data/json'
    Model_save_dir = 'data/models'
    Optimizable_parameters_f_name = 'optimizable_parameters.json'

    def __init__(self, sk_model):
        self._model = sk_model
        self._model_name = self._model.__name__
        self._model_name_lower = self._model_name.lower()
        self._is_optimized = False
        self._optimal_params = {}
        self._dm = DataManager()

        f = os.path.join(
            Model.Optimizable_parameters_path,
            Model.Optimizable_parameters_dir,
            Model.Optimizable_parameters_f_name
        )
        with open(f, 'r') as params:
            parameters = json.load(params)

        optimizable_parameters = parameters[self._model_name]["optimizable_parameters"]
        if('RandomSearch' in optimizable_parameters.keys()) :
            self._ramdom_opt_params = optimizable_parameters['RandomSearch']
        else :
            self._ramdom_opt_params = {}
        if('GridSearch' in optimizable_parameters.keys()) :
            self._grid_opt_params = optimizable_parameters['GridSearch']
        else :
            self._grid_opt_params = {}
        
        self._fixed_parameters = parameters[self._model_name]["fixed_parameters"]
        self.build_pipeline()

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
        if self._is_optimized:
            return self._optimal_params
        else:
            raise ValueError(
                'Model has not been optimized, could not find optimal values.')

    def get_optimal_model(self):
        if self._is_optimized:
            return self._pipeline
        else:
            raise ValueError('Model has not been optimized.')

    def update_model(self) :
        self._pipeline.set_params(**self._optimal_params)

    def optimize_model(self):
        print(f'Optimizing model : {self._model_name}...')
        if self._pipeline != None:
            mo = ModelOptimizer(self)
            # Checking that the dict is not empty
            if self._ramdom_opt_params :
                self._optimal_params.update(
                    mo.random_search_optimize(
                        self._dm._train_X,
                        self._dm._train_y, 
                        self._ramdom_opt_params
                    )
                )
                self.update_model()
            # Checking that the dict is not empty
            if self._grid_opt_params :
                self._optimal_params.update(
                    mo.grid_search_optimize(
                        self._dm._train_X,
                        self._dm._train_y,
                        self._grid_opt_params
                    )
                )
                self.update_model()
            self._is_optimized = True
            print('Optimization finished.')
        else:
            raise ValueError('No pipeline is set up, nothing to optimize.')

    def save_model(self) :
        model_params = {
            'fixed' : self._fixed_parameters,
            'opt' : self._optimal_params
        }
        
        with open(os.path.join(
            Model.Optimizable_parameters_path,
            Model.Model_save_dir,
            self._model_name_lower + '.model'
        ), 'w') as f:
            json.dump(model_params, f)

    def fit(self) :
        print(f'Fitting training data for model {self._model_name}...')
        self._pipeline.fit(X = self._dm._train_X, y = self._dm._train_y)
        print(self._pipeline.score(X = self._dm._test_X, y = self._dm._test_y))
