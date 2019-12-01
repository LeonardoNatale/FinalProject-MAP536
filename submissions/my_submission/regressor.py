from sklearn.base import BaseEstimator
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from Model.model import Model
from Service.data_manager import DataManager

dm = DataManager(ramp=True)


class Regressor(BaseEstimator):
    def __init__(self):
        # m = Model(HistGradientBoostingRegressor)
        m = Model(HistGradientBoostingRegressor, from_file=True)
        m.load_from_file()
        self.reg = AdaBoostRegressor(m.get_pipeline())

    def fit(self, X, y):
        print("fit...")
        X = dm.append_to_data(X)
        self.reg.fit(X, y)

    def predict(self, X):
        X = dm.append_to_data(X)
        return self.reg.predict(X)
