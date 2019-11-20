import inspect
import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

from Service.data_manager import DataManager
from Model.model import Model


class MultiModel:

    def __init__(self, models):
        print('Initializing mulitimodel...')
        self._dm = DataManager()
        self._models = []
        init_str = ''
        if isinstance(models, dict):
            if 'load_from_file' in models.keys():
                lff = models['load_from_file']
            else:
                lff = False
            init_str = [mod.__name__ for mod in models['models'].keys()]
            for model, params in models['models'].items():
                self._models.append(
                    Model(
                        model,
                        self._dm,
                        params['fixed_parameters'],
                        params['optimizable_parameters']
                    )
                )
            if lff:
                self._models[-1].load_from_file()
        # Otherwise check if it's only a sklearn module.
        elif inspect.getmodule(models).__name__.split('.')[0] == 'sklearn':
            init_str = models.__name__
            self._models.append(Model(models, self._dm))
        print(f"Initialized a multimodel with the following models : {init_str}")

    # TODO rewrite.
    def add(self, model):
        if inspect.getmodule(model).__name__.split('.')[0] == 'sklearn':
            print(f'Adding model {model.__name__}')
            self._models.append(Model(model))
        else:
            raise TypeError('Parameter model should be an sklearn model, {model.__name__} given')

    def optimize(self, force=False):
        for model in self._models:
            model.optimize_model(force=force)

    def get_optimal_parameters(self):
        opt_params = {}
        for model in self._models:
            opt_params[model.model_name] = model.get_optimal_parameters()
        return opt_params

    def fit(self):
        for model in self._models:
            model.fit()

    def are_all_optimized(self):
        for model in self._models:
            if not model.is_optimized:
                return False
        return True

    def check_all_optimized(self):
        if not self.are_all_optimized():
            raise ValueError(f'All the models are mot optimized, please run optimize().')

    def predict(self, reduced_pred=True):
        # Checking that every model is optimized before doing any prediction :
        self.check_all_optimized()
        df = pd.DataFrame(
            columns=[mod.model_name for mod in self._models],
            index=range(self._models[0].dm.get_test_X().shape[0])
        )
        for model in self._models:
            pred = model.predict()
            if pred is not None:
                df.loc[:, model.model_name] = pred
            else:
                raise ValueError(f'The model {model.model_name} has not been fitted yet, could not do prediction.')
        if reduced_pred:
            df = df.mean(axis=1)
        return df

    def score(self, reduced_pred=True):
        self.check_all_optimized()
        pred = self.predict(reduced_pred=reduced_pred)
        true = self._models[0].dm.get_test_y()
        if reduced_pred:
            data = pred - true
        else:
            data = pred.subtract(pd.Series(true), axis=0)
        return np.sqrt(np.mean(data ** 2))


# Test
GradientBoostingRegressor()

# multi = MultiModel(
#     {
#         "models": {
#             Ridge: {
#                 "fixed_parameters": {
#                     "tol": 1e-3
#                 },
#                 "optimizable_parameters": {
#                     "RandomSearch": {
#                         "alpha": scipy.stats.uniform(1, 20)
#                     },
#                     "GridSearch": {
#                         "solver": ["svd", "cholesky", "lsqr"]
#                     }
#                 }
#             },
#             GradientBoostingRegressor: {
#                 "fixed_parameters": {
#                     "loss": 'ls'
#                 },
#                 "optimizable_parameters": {
#                     "RandomSearch": {
#                         "alpha": scipy.stats.uniform(0, 1),
#                         "max_depth": scipy.stats.uniform(2, 10)
#                     },
#                     "GridSearch": {
#                         "criterion": ["friedman_mse", "mse"]
#                     }
#                 }
#             }
#
#         }
#     }
# )

# multi = MultiModel(
#     {
#         "models": {
#             GradientBoostingRegressor: {
#                 "fixed_parameters": {
#                     "loss": 'ls'
#                 },
#                 "optimizable_parameters": {
#                     "RandomSearch": {
#                         "alpha": scipy.stats.uniform(0, 1),
#                         "max_depth": scipy.stats.uniform(2, 10)
#                     },
#                     "GridSearch": {
#                         "criterion": ["friedman_mse", "mse"]
#                     }
#                 }
#             }
#
#         }
#     }
# )

multi = MultiModel(
    {
        "models": {
            HistGradientBoostingRegressor: {
                "fixed_parameters": {
                    "loss": 'least_squares'
                },
                "optimizable_parameters": {
                    "RandomSearch": {
                        "learning_rate": scipy.stats.uniform(0, 1),
                        "l2_regularization": scipy.stats.uniform(0, 1)
                    }
                }
            }

        }
    }
)

multi.optimize()
multi.fit()

print(multi.score(reduced_pred=False))
print(multi.score(reduced_pred=True))
