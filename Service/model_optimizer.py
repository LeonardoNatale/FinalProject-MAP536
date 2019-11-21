import json
import os
import scipy.stats
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# TODO Would be nice to create a Model class which contains
# the model and uses this class to optimize it.
# I.e. get rid of make_pipeline method and create a specific class.

class ModelOptimizer:
    Optimizable_parameters_path = '.'
    Optimizable_parameters_dir = 'data/json'
    Optimizable_parameters_f_name = 'optimizable_parameters.json'

    def __init__(self, model):
        self._model = model
        self._pipeline = model._pipeline

    def grid_search_optimize(self, X, y, grid_opt_param):
        # TODO update documentation
        """
        For every parameter specified in the `optimizable_parameters` field
        of the json, we create a RandomizedSearchCV with the given params
        to which we associate the given distribution.
        The output is the optimal value of the parameters for the model.
        """
        grid_params = {
            self._model._model_name_lower + '__' + str(key): val
            for key, val in grid_opt_param.items()
        }
        print('Beginning Grid search for the following parameters : ')
        print(grid_params)
        model_grid_search = GridSearchCV(
            self._pipeline,
            param_grid=grid_params,
            n_jobs=4,
            cv=5
        )
        model_grid_search.fit(X, y)
        return model_grid_search.best_params_

    def random_search_optimize(self, X, y, random_opt_param):
        # TODO update documentation
        """
        For every parameter specified in the `optimizable_parameters` field
        of the json, we create a RandomizedSearchCV with the given params
        to which we associate the given distribution.
        The output is the optimal value of the parameters for the model.
        """
        distributions = {
            self._model._model_name_lower + '__' + str(key): val
            for key, val in random_opt_param.items()
        }
        print('Beginning Randomized search for the following parameters : ')
        print(list(distributions.keys()))
        model_random_search = RandomizedSearchCV(
            self._pipeline,
            param_distributions=distributions,
            n_jobs=-1,
            cv=5
        )
        model_random_search.fit(X, y)
        # Merging the optimized values into one single dict.
        return model_random_search.best_params_
