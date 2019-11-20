from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

from Model.model import Model


class Regressor(BaseEstimator):
    def __init__(self):
        m = Model(HistGradientBoostingRegressor)
        m.load_from_file()
        self.reg = m._pipeline

    def fit(self, X, y):
        self.reg.fit(X, y)


    def predict(self, X):
        return self.reg.predict(X)
