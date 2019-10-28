import json
import os
import scipy.stats
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

# TODO Would be nice to create a Model class which contains
# the model and uses this class to optimize it.
# I.e. get rid of make_pipeline method and create a specific class.

class ModelOptimizer():

    Optimizable_parameters_path = '.'
    Optimizable_parameters_dir = 'data/json'
    Optimizable_parameters_f_name = 'optimizable_parameters.json'
    
    def __init__(self, model):
        self._model = model
        self._pipeline = model._pipeline
        
    def CV(self, X, y) :
        """
        For every parameter specified in the `optimizable_parameters` field
        of the json, we create a RandomizedSearchCV with the given params
        to which we associate the given distribution.
        The output is the optimal value of the parameters for the model.
        """
        distributions = {}
        opt_param = self._model.get_optimizable_parameters()
        for key in opt_param.keys():
            law = opt_param[key]['law']
            kwg = opt_param[key]['kwargs']
            distributions[self._model._model_name_lower + '__' + key] = getattr(scipy.stats, law)(**kwg)
        print('Beginning Randomized search for the following parameters : ')
        print(list(distributions.keys()))
        model_grid_search = RandomizedSearchCV(
            self._pipeline,
            param_distributions=distributions, 
            n_iter=3,
            n_jobs=4, 
            cv=5
        )
        model_grid_search.fit(X, y)
        return model_grid_search.best_params_
