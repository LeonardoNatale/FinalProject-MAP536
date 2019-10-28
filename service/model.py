import os, json
from sklearn.pipeline import Pipeline
from model_optimizer import ModelOptimizer

class Model():

    Optimizable_parameters_path = '.'
    Optimizable_parameters_dir = 'data/json'
    Optimizable_parameters_f_name = 'optimizable_parameters.json'
    
    def __init__(self, sk_model):
        self._model = sk_model
        self._model_name = self._model.__name__
        self._model_name_lower = self._model_name.lower()
        print(f'Model : {self._model}\nName : {self._model_name}')
        self._is_optimized = False
        self._optimal_params = None

        f = os.path.join(
            Model.Optimizable_parameters_path,
            Model.Optimizable_parameters_dir,
            Model.Optimizable_parameters_f_name
        )
        with open(f, 'r') as params:
            parameters = json.load(params)
        
        self._optimizable_parameters = parameters[self._model_name]["optimizable_parameters"]
        self._fixed_parameters = parameters[self._model_name]["fixed_parameters"]
        self.build_pipeline()
        
    def get_optimizable_parameters(self):
        return self._optimizable_parameters

    def get_fixed_parameters(self):
        return self._fixed_parameters
    
    def build_pipeline(self) :
        self._pipeline = Pipeline(
            [
                (
                    self._model_name_lower,
                    self._model(**self._fixed_parameters)
                )
            ]
        )    
    
    def get_optimal_parameters(self):
        if self._is_optimized :
            return self._optimal_params
        else :
            raise ValueError('Model has not been optimized, could not find optimal values.')

    def optimize_model(self):
        if self._pipeline != None :
            mo = ModelOptimizer(self)
            self._optimal_params = mo.CV()
            self._pipeline.set_params(**self._optimal_params)
            self._is_optimized = True
        else:
            raise ValueError('No pipeline is set up, nothing to optimize.')
