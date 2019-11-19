import json
from model import Model

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

x = Model(Ridge)

x.fit()

x.optimize_model()
print(x.get_optimal_parameters())
print(x.get_optimal_model())

x.fit()
x.r2_score()
print(x.predict())

# x.save_model()


# Takes long to run
# x = Model(SVR)
# x.optimize_model()
# print(x.get_optimal_parameters())