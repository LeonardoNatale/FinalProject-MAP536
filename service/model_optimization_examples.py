from model import Model

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

x = Model(RandomForestRegressor)
x.optimize_model()
print(x.get_optimal_parameters())

x = Model(SVR)
x.optimize_model()
print(x.get_optimal_parameters())