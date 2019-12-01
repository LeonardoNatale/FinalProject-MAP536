import inspect
import pandas as pd
import numpy as np
from Service.data_manager import DataManager
from Model.model import Model


class MultiModel:
    """
    This class is used to fit multiple models at once and group them.
    """

    def __init__(self, models):
        """
        :param models: either a sklearn model or a dictionary of models and their parameters.
        """
        print('Initializing MultiModel...')
        self._dm = DataManager()
        self._models = []
        init_str = ''
        if isinstance(models, dict):
            if 'load_from_file' in models.keys():
                lff = models['load_from_file']
            else:
                lff = False
            init_str = [mod.__name__ for mod in models['models'].keys()]

            for model, params in models['models'].items():
                self._models.append(
                    Model(
                        model,
                        self._dm,
                        params['fixed_parameters'],
                        params['optimizable_parameters']
                    )
                )

            if lff:
                self._models[-1].load_from_file()
        # Otherwise check if it's only a sklearn module.
        elif inspect.getmodule(models).__name__.split('.')[0] == 'sklearn':
            init_str = models.__name__
            self._models.append(Model(models, self._dm))
        print(f"Initialized a MultiModel with the following models : {init_str}")

    def add(self, model):
        # TODO add the ability to add an already optimized model.
        """
        Adds an sklearn model to the MultiModel

        :param model: The name of the sklearn model.
        """
        if inspect.getmodule(model).__name__.split('.')[0] == 'sklearn':
            print(f'Adding model {model.__name__}')
            self._models.append(Model(model), self._dm)
        else:
            raise TypeError('Parameter model should be an sklearn model, {model.__name__} given')

    def optimize(self, force=False):
        """
        Optimizes all the models sequentially.

        :param force: True to force the model to optimize even though
        it's already optimized.
        """
        for model in self._models:
            model.optimize_model(force=force)

    def get_optimal_parameters(self):
        """
        Returns the optimal parameters of all the models as a dictionary.
        :return: A dictionary of the optimal parameters.
        """
        opt_params = {}
        for model in self._models:
            opt_params[model.model_name] = model.get_optimal_parameters()
        return opt_params

    def fit(self):
        """
        Fits all the models.
        """
        for model in self._models:
            model.fit()

    def are_all_optimized(self):
        """
        Check whether all the models are optimized.
        :return:
        """
        for model in self._models:
            if not model.is_optimized:
                return False
        return True

    def check_all_optimized(self):
        """
        Raises a ValueError if all the models aren't optimized.
        :return:
        """
        if not self.are_all_optimized():
            raise ValueError(f'All the models are mot optimized, please run optimize().')

    def predict(self, reduced_pred=True):
        """
        Predicts on new data. Aggregates the result as a average if `reduced_pred` is True.
        :param reduced_pred: If true, the prediction is the average of the prediction of all the models.
        :return: The prediction as a DataFrame.
        """
        # Checking that every model is optimized before doing any prediction :
        self.check_all_optimized()
        df = pd.DataFrame(
            columns=[mod.model_name for mod in self._models],
            index=range(self._models[0].dm.get_test_X().shape[0])
        )
        for model in self._models:
            pred = model.predict()
            if pred is not None:
                df.loc[:, model.model_name] = pred
            else:
                raise ValueError(f'The model {model.model_name} has not been fitted yet, could not do prediction.')
        if reduced_pred:
            df = df.mean(axis=1)
        return df

    def score(self, reduced_pred=True):
        """
        Computes the RMSE of the models, computes the average if reduced_pred is True.
        :param reduced_pred: If true, the RMSE is the average of the RMSE of all the models.
        :return: The RMSE.
        """
        self.check_all_optimized()
        pred = self.predict(reduced_pred=reduced_pred)
        true = self._models[0].dm.get_test_y()
        if reduced_pred:
            data = pred - true
        else:
            data = pred.subtract(pd.Series(true), axis=0)
        return np.sqrt(np.mean(data ** 2))

    def multi_model_testing(self):
        """
        Utility function to compare model one to another.
        :return: A DataFrame with the models performances.
        """
        cols = ['model_name', 'opt_params', 'rmse_before', 'rmse_after', 'fit_time']
        df = pd.DataFrame(columns=cols)
        for model in self._models:
            df = df.append(pd.Series(model.model_quality_testing(as_df=True), index=cols), ignore_index=True)
        return df
