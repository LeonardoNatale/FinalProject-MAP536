from Model.multi_model import MultiModel
from Model.model import Model
from Service.external_data_generator import ExternalDataGenerator
import scipy.stats
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
from Service.data_manager import DataManager
from Service.ramp_external_data_generator import RampExternalDataGenerator
from Model.ramp_model import RampModel

model = True
multi = False
external_data = False
data_manager = False

# ------------------ DATA MANAGER ------------------ #

if data_manager:
    dm = DataManager()
    dm.get_train_X().head(10)

# ------------------ EXTERNAL DATA ------------------ #

# To generate external data once again
if external_data:
    x = RampExternalDataGenerator()
    x._write_external_data()

# ------------------ MODELS ------------------ #

if model:

    model_type = HistGradientBoostingRegressor
    fixed = {"loss": 'least_squares'}
    opt = {
        "RandomSearch": {
            "l2_regularization": scipy.stats.uniform(0, 10),
            "min_samples_leaf": scipy.stats.randint(low=10, high=50),
            "max_iter": scipy.stats.randint(low=50, high=200)
        }
    }

    x = RampModel(model_type, fixed_parameters=fixed, optimizable_parameters=opt)
    dm = x.get_data_manager()
    x.model_quality_testing(
        train_x=dm.get_train_X(),
        train_y=dm.get_train_y(),
        test_x=dm.get_test_X(),
        test_y=dm.get_test_y()
    )

    # model_type = GradientBoostingRegressor
    # fixed = {}
    # opt = {
    #     "RandomSearch": {
    #         "learning_rate": scipy.stats.uniform(0, 1),
    #         "max_features": scipy.stats.uniform(0, 1),
    #         "alpha": scipy.stats.uniform(0, 1)
    #     },
    #     'GridSearch': {
    #         "loss": ['ls', 'quantile'],
    #         "criterion": ["friedman_mse", "mse"]
    #     }
    # }
    #
    # opt = {
    #     "RandomSearch": {},
    #     'GridSearch': {}
    # }

    # model_type = BaggingRegressor
    # fixed = {
    #     "base_estimator": HistGradientBoostingRegressor(),
    #     "n_jobs": -1
    # }
    # opt = {
    #     "RandomSearch": {},
    #     "GridSearch": {}
    # }

    # model_type = AdaBoostRegressor
    # fixed = {
    #     "base_estimator": HistGradientBoostingRegressor(),
    # }
    # opt = {
    #     "RandomSearch": {},
    #     "GridSearch": {}
    # }

    # x = Model(model_type, fixed_parameters=fixed, optimizable_parameters=opt)
    #
    # x.model_quality_testing()
    # x.feature_importance()
    # x.save_model()


# ------------------ MULTI MODELS ------------------ #

# if multi:
    # multi = MultiModel(
    #     {
    #         "models": {
    #             GradientBoostingRegressor: {
    #                 "fixed_parameters": {
    #                     "loss": 'ls'
    #                 },
    #                 "optimizable_parameters": {
    #                     "RandomSearch": {
    #                         "alpha": scipy.stats.uniform(0, 1),
    #                         "max_depth": scipy.stats.uniform(2, 10)
    #                     },
    #                     "GridSearch": {
    #                         "criterion": ["friedman_mse", "mse"]
    #                     }
    #                 }
    #             }
    #         }
    #     }
    # )
    #
    # multi = MultiModel(
    #     {
    #         "models": {
    #             HistGradientBoostingRegressor: {
    #                 "fixed_parameters": {
    #                     "loss": 'least_squares'
    #                 },
    #                 "optimizable_parameters": {
    #                     "RandomSearch": {
    #                         "learning_rate": scipy.stats.uniform(0, 1),
    #                         "l2_regularization": scipy.stats.uniform(0, 1)
    #                     }
    #                 }
    #             }
                # ,
                # GradientBoostingRegressor: {
                #     "fixed_parameters": {
                #         "loss": 'ls'
                #     },
                #     "optimizable_parameters": {
                #         "RandomSearch": {
                #             "alpha": scipy.stats.uniform(0, 1),
                #             "max_depth": scipy.stats.uniform(2, 10)
                #         },
                #         "GridSearch": {
                #             "criterion": ["friedman_mse", "mse"]
                #         }
                #     }
                # }
    #         }
    #     }
    # )
    #
    # print(multi.multi_model_testing())


# model_type = AdaBoostRegressor
# fixed = {
#     "base_estimator": HistGradientBoostingRegressor(
#         l2_regularization=0.9752299302272766,
#         learning_rate=0.153187560120574
#     ),
# }

