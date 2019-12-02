from sklearn.pipeline import Pipeline
from Service.ramp_data_manager import RampDataManager
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class RampModel:

    def __init__(self, sk_model, dm=None, fixed_parameters=None, optimizable_parameters=None):
        self._model = sk_model
        self.model_name = self._model.__name__
        self._model_name_lower = self.model_name.lower()
        self.__dm = RampDataManager()
        self.__fixed_parameters = fixed_parameters
        self.__optimizable_parameters = optimizable_parameters
        self.__pipeline = Pipeline(
            [
                (
                    self._model_name_lower,
                    self._model(**self.__fixed_parameters)
                )
            ]
        )

    def get_pipeline(self):
        return self.__pipeline

    def fit(self, x, y):
        print('Fitting training data...')
        new_x = self.__dm.append_external_data(x)
        print(f'X columns :\n{list(new_x.columns)}')
        self.__pipeline.fit(X=new_x, y=y)

    def predict(self, x):
        print('Making prediction...')
        new_x = self.__dm.append_external_data(x)
        return self.__pipeline.predict(X=new_x)

    def rmse(self, x, y):
        """
        Returns the RMSE of the model on test data.

        :return: The RMSE of the model.
        """
        return np.sqrt(mean_squared_error(self.predict(x), y))

    def feature_importance(self, x):
        ordering = np.argsort(self.__pipeline[0].feature_importances_)[::-1][:20]
        importances = self.__pipeline[0].feature_importances_[ordering]
        feature_names = self.__dm.append_external_data(x).columns[ordering]
        x = np.arange(len(feature_names))
        plt.figure()
        plt.bar(x, importances)
        plt.xticks(x, feature_names, rotation=90)
        plt.show()

