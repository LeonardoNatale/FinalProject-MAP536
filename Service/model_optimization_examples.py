import json
from Model.model import Model
import scipy.stats
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.linear_model import SGDRegressor




def make_process(x, model_type):
    x.fit()

    x.optimize_model()
    print(x.get_optimal_parameters())
    print(x.get_optimal_model())

    x.fit()
    x.r2_score()
    print(x.predict())

    x.save_model()

    x = Model(model_type)
    x.load_from_file()
    print(x.get_optimal_parameters())


model_type = HistGradientBoostingRegressor
x = Model(
    model_type,
    fixed_parameters={"loss": 'least_squares'},
    optimizable_parameters={
        "RandomSearch": {
            "learning_rate": scipy.stats.uniform(0, 1),
            "l2_regularization": scipy.stats.uniform(0, 1)
        }
    }
)

# make_process(x, model_type)

model_type = GradientBoostingRegressor
x = Model(
    model_type,
    fixed_parameters={},
    optimizable_parameters={
        "RandomSearch": {
            "learning_rate": scipy.stats.uniform(0, 1),
            "max_features": scipy.stats.uniform(0, 1),
            "alpha": scipy.stats.uniform(0, 1)
        },
        'GridSearch': {
            "loss" : ['ls', 'quantile'],
            "criterion": ["friedman_mse", "mse"]
        }
    }
)

make_process(x, model_type)
