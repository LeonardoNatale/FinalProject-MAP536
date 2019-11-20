from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
import pandas as pd

from Model.model import Model


class Regressor(BaseEstimator):
    def __init__(self):
        m = Model(Ridge)
        m.load_from_file()
        self.reg = {'R' : m._pipeline}

    def fit(self, X, y):
        for regressor in self.reg.values():
            regressor.fit(X, y)


    def predict(self, X):
        prediction = pd.DataFrame(columns = self.reg.keys())
        for regressor_name in self.reg.keys() :
            prediction.loc[:, regressor_name] = self.reg[regressor_name].predict(X)
        prediction_vector = prediction.mean(axis = 1)
        print(prediction.head())
        print(prediction_vector)
        return prediction_vector