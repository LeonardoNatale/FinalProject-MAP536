from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


class ModelOptimizer:
    """
    This class is in charge of the optimization of the models by GridSearch and RandomizedSearch
    """
    def __init__(self, model):
        self._model = model

    def grid_search_optimize(self, x, y, grid_opt_param):
        """
        For every parameter specified in grid_opt_param, we create
        a GridSearchCV with the given params to which we associate
        the given distribution. The output is the optimal value of the
        parameters for the model.
        """
        if grid_opt_param:
            grid_params = {
                self._model.get_model_name_lower() + '__' + str(key): val
                for key, val in grid_opt_param.items()
            }
            print(f'Beginning Grid search for the following parameters : {list(grid_params)}')
            model_grid_search = GridSearchCV(
                self._model.get_pipeline(),
                param_grid=grid_params,
                n_jobs=4,
                cv=5
            )
            model_grid_search.fit(x, y)
            return model_grid_search.best_params_
        else:
            return {}

    def random_search_optimize(self, x, y, random_opt_param):
        """
        For every parameter specified in random_opt_param, we create
        a RandomSearchCV with the given params to which we associate
        the given distribution. The output is the optimal value of the
        parameters for the model.
        """
        if random_opt_param:
            distributions = {
                self._model.get_model_name_lower() + '__' + str(key): val
                for key, val in random_opt_param.items()
            }
            print(f'Beginning Randomized search for the following parameters : {list(distributions.keys())}')
            model_random_search = RandomizedSearchCV(
                self._model.get_pipeline(),
                param_distributions=distributions,
                n_jobs=-1,
                cv=5
            )
            model_random_search.fit(x, y)
            # Merging the optimized values into one single dict.
            return model_random_search.best_params_
        else:
            return {}