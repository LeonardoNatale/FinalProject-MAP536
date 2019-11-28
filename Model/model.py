import os
import json
import numpy as np
from sklearn.pipeline import Pipeline
from Service.model_optimizer import ModelOptimizer
from Service.data_manager import DataManager
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error


class Model:
    """
    The objective of this class is to contain utility methods for optimizing and testing models.
    It contains an sklearn model stored as a Pipeline, which is optimized through the process.
    """

    # Locations of the data.
    Optimizable_parameters_path = '.'
    Optimizable_parameters_dir = 'data/json'
    Model_save_dir = 'data/models'
    Optimizable_parameters_f_name = 'optimizable_parameters.json'

    def __init__(self, sk_model, dm=DataManager(), fixed_parameters=None, optimizable_parameters=None):
        """
        Constructor of the model.

        :param sk_model: Sklearn model to implement.
        :param dm: DataManager instance used to retrieve the data.
        :param fixed_parameters: Constant parameters of the model, given as a dictionary
        :param optimizable_parameters: Parameters to be optimized by this class, given as a dictionary.
        """
        # Storing the model type
        self._model = sk_model
        self.model_name = self._model.__name__
        self._model_name_lower = self.model_name.lower()
        # Data about optimization
        self.is_optimized = False
        self._optimal_params = {}
        # Data objects
        self.dm = dm
        self._pipeline = None

        # Parameter objects
        self._fixed_parameters = dict() if fixed_parameters is None else fixed_parameters
        self._random_opt_params = {}
        # If no parameter is provided, then we use an empty dict.
        if 'RandomSearch' in optimizable_parameters.keys():
            self._random_opt_params = optimizable_parameters['RandomSearch']
        self._grid_opt_params = {}
        if 'GridSearch' in optimizable_parameters.keys():
            self._grid_opt_params = optimizable_parameters['GridSearch']
        self.build_pipeline()

    def get_model_name_lower(self):
        return self._model_name_lower

    def get_pipeline(self):
        return self._pipeline

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

    def fit(self):
        """
        Fits the model with the given parameters. If the parameters aren't optimized, a warning is given.
        """
        if not self.is_optimized:
            print(f'Running a fit on a non optimized model : {self.model_name}. Consider running optimize_model() for '
                  f'better performances.')
        print(f'Fitting training data for model {self.model_name}...')
        self._pipeline.fit(X=self.dm.get_train_X(), y=self.dm.get_train_y())

    def predict(self):
        """
        Predicts the test data with the fitted model.
        If the model is not fitted, this throws an error.
        :return: The prediction of the model.
        """
        try:
            prediction = self._pipeline.predict(self.dm.get_test_X())
        except NotFittedError as e:
            print(f'Model {self.model_name} not fitted. Please run fit().\n{e}')
            raise ValueError(f'Model {self.model_name} not fitted. Please run fit().')
        return prediction

    def r2_score(self):
        """
        Computes the R^2 coefficient of the model given the test data.

        :return: The R^2 coefficient of the model.
        """
        return self._pipeline.score(
            X=self.dm.get_test_X(),
            y=self.dm.get_test_y()
        )

    def rmse(self):
        """
        Returns the RMSE of the model on test data.

        :return: The RMSE of the model.
        """
        return np.sqrt(mean_squared_error(self.predict(), self.dm.get_test_y()))

    def model_quality_testing(self, as_df=False):
        """
        Utility function that computes the difference of RMSE before and after optimization.

        :param as_df: Whereas the result should be returned as a DataFrame or just printed to the console.
        :return: A DataFrame with the before/after RMSEs, the model name and it's optimized parameters.
        """
        self.fit()
        before_rmse = self.rmse()
        self.optimize_model()
        self.fit()
        if as_df:
            return [self._model_name_lower, self.get_optimal_parameters(), before_rmse, self.rmse()]
        else:
            print(f'Optimal parameters of the model :\n{self.get_optimal_parameters()}')
            print(f'RMSE of non optimized model : {before_rmse}')
            print(f'RMSE of optimized model : {self.rmse()}')

    def save_model(self):
        """
        Saves the model to the 'self._model_name_lower.model' file.
        """

        model_params = {
            'fixed': self._fixed_parameters,
            'opt': self._optimal_params
        }

        with open(os.path.join(
                Model.Optimizable_parameters_path,
                Model.Model_save_dir,
                self._model_name_lower + '.model'
        ), 'w') as f:
            json.dump(model_params, f)

    def load_from_file(self):
        """
        If this method is called, then the model is going to be replaced
        by a model saved in a file in the './data/models/'
        The corresponding file is 'self._model_name_lower.model'
        """
        # Reading the file.
        with open(os.path.join(
                Model.Optimizable_parameters_path,
                Model.Model_save_dir,
                self._model_name_lower + '.model'
        ), 'r') as f:
            params = json.load(f)

        # Updating the parameters to the one of the file.
        self._fixed_parameters = params['fixed']
        self.build_pipeline()
        self._optimal_params = params['opt']
        self.update_model()
        self.is_optimized = True
