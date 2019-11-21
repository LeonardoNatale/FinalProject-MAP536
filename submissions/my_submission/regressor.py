from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
import os
from Model.model import Model
from Service.data_manager import DataManager


class Regressor(BaseEstimator):
    def __init__(self):
        print(os.listdir("./../"))
        m = Model(HistGradientBoostingRegressor)
        m.load_from_file()
        self.dm = DataManager(ramp=True)
        self.reg = m.get_pipeline()

    def fit(self, X, y):
        X = self.dm.append_to_data(X)
        self.reg.fit(X, y)

    def predict(self, X):
        X = self.dm.append_to_data(X)
        return self.reg.predict(X)
