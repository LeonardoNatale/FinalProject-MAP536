import json
from Model.model import Model
import scipy.stats
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.linear_model import SGDRegressor


x = Model(
    HistGradientBoostingRegressor,
    fixed_parameters={"loss": 'least_squares'},
    optimizable_parameters={
        "RandomSearch": {
            "learning_rate": scipy.stats.uniform(0, 1),
            "l2_regularization": scipy.stats.uniform(0, 1)
        }
    }
)

# x.model_quality_testing()

x = Model(
    GradientBoostingRegressor,
    fixed_parameters={},
    optimizable_parameters={
        "RandomSearch": {
            "learning_rate": scipy.stats.uniform(0, 1),
            "max_features": scipy.stats.uniform(0, 1),
            "alpha": scipy.stats.uniform(0, 1)
        },
        'GridSearch': {
            "loss": ['ls', 'quantile'],
            "criterion": ["friedman_mse", "mse"]
        }
    }
)

x.model_quality_testing()