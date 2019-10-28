from model import Model

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

x = Model(RandomForestRegressor)
x.optimize_model()
print(x.get_optimal_parameters())
# Here we can see that the model that get printed has the optimal values.
print(x.get_optimal_model())

# Takes long to run
# x = Model(SVR)
# x.optimize_model()
# print(x.get_optimal_parameters())