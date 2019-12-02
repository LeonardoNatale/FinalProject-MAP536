from sklearn.pipeline import Pipeline
from Service.ramp_data_manager import RampDataManager
import numpy as np
from sklearn.metrics import mean_squared_error
from Service.model_optimizer import ModelOptimizer
import matplotlib.pyplot as plt


class RampModel:

    def __init__(self, sk_model, dm=None, fixed_parameters=None, optimizable_parameters=None):
        self._model = sk_model
        self.model_name = self._model.__name__
        self._model_name_lower = self.model_name.lower()
        self.__dm = RampDataManager()
        self.is_optimized = False
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

    def get_fixed_parameters(self):
        return self._fixed_parameters

    def get_optimal_parameters(self):
        if self.is_optimized:
            return self._optimal_params
        else:
            raise ValueError('Model has not been optimized. Please run optimize_model()')

    def get_optimal_model(self):
        if self.is_optimized:
            return self._pipeline
        else:
            raise ValueError('Model has not been optimized. Please run optimize_model()')

    def build_pipeline(self):
        """
        Creates a pipeline with the model and its fixed parameters.
        """
        self._pipeline = Pipeline(
            [
                (
                    self._model_name_lower,
                    self._model(**self._fixed_parameters)
                )
            ]
        )

    def optimize_model(self, force=False):
        """
        Optimizes the model using GridSearch and RandomSearch and updates the model accordingly.
        :param force: Boolean to force re-optimization if the model is already optimized.
        """

        if self.is_optimized and not force:
            print(
                f'Model {self.model_name} is already optimized, doing nothing, use force = True to force '
                f're-optimization.')
        else:
            print(f'Optimizing model : {self.model_name}...')
            # Check that we have a pipeline, which should always be the case.
            if self._pipeline is not None:
                mo = ModelOptimizer(self)
                # Checking that the dict is not empty
                if self._random_opt_params:
                    # Doing the RandomSearchCV
                    self._optimal_params.update(
                        mo.random_search_optimize(
                            self.dm.get_train_X(),
                            self.dm.get_train_y(),
                            self._random_opt_params
                        )
                    )
                    # Updating model.
                    self.update_model()
                # Checking that the dict is not empty
                if self._grid_opt_params:
                    # Doing the GridSearchCV
                    self._optimal_params.update(
                        mo.grid_search_optimize(
                            self.dm.get_train_X(),
                            self.dm.get_train_y(),
                            self._grid_opt_params
                        )
                    )
                    # Updating model.
                    self.update_model()
                self.is_optimized = True
                print('Optimization finished.')
            else:
                raise ValueError('No pipeline is set up, nothing to optimize.')

    def update_model(self):
        """
        Updates the model with the most recent optimized parameters.
        """
        self._pipeline.set_params(**self._optimal_params)

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

