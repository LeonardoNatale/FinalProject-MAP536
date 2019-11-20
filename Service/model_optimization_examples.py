import json
from Model.model import Model
import scipy.stats
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

# x = Model(RandomForestRegressor)
# x.optimize_model()
# print(x.get_optimal_parameters())
# # Here we can see that the model that get printed has the optimal values.
# print(x.get_optimal_model())
# x.save_model()

# t = Model(RandomForestRegressor)
# t.load_from_file()
# print(t._pipeline)
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

x.fit()

x.optimize_model()
print(x.get_optimal_parameters())
print(x.get_optimal_model())

x.fit()
x.r2_score()
print(x.predict())

x.save_model()

x = Model(HistGradientBoostingRegressor)
x.load_from_file()
print(x.get_optimal_parameters())
