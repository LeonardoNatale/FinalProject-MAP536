from Model.multi_model import MultiModel
from Model.model import Model
from Service.external_data_generator import ExternalDataGenerator
import scipy.stats
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
from Service.data_manager import DataManager

model = False
multi = False
external_data = False

dm = DataManager()
# ------------------ EXTERNAL DATA ------------------ #

# To generate external data once again
if external_data:
    x = ExternalDataGenerator()
    x._write_external_data()

# ------------------ MODELS ------------------ #

if model:
    x = Model(
        HistGradientBoostingRegressor,
        fixed_parameters={"loss": 'least_squares'},
        optimizable_parameters={
            "RandomSearch": {
                "learning_rate": scipy.stats.uniform(0, 1),
                "l2_regularization": scipy.stats.uniform(0, 1)
            }
        }
    )

    # x.model_quality_testing()

    x = Model(
        GradientBoostingRegressor,
        fixed_parameters={},
        optimizable_parameters={
            "RandomSearch": {
                "learning_rate": scipy.stats.uniform(0, 1),
                "max_features": scipy.stats.uniform(0, 1),
                "alpha": scipy.stats.uniform(0, 1)
            },
            'GridSearch': {
                "loss": ['ls', 'quantile'],
                "criterion": ["friedman_mse", "mse"]
            }
        }
    )

    x.model_quality_testing()

# ------------------ MULTI MODELS ------------------ #

if multi:
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

    multi = MultiModel(
        {
            "models": {
                HistGradientBoostingRegressor: {
                    "fixed_parameters": {
                        "loss": 'least_squares'
                    },
                    "optimizable_parameters": {
                        "RandomSearch": {
                            "learning_rate": scipy.stats.uniform(0, 1),
                            "l2_regularization": scipy.stats.uniform(0, 1)
                        }
                    }
                }
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
            }
        }
    )

    print(multi.multi_model_testing())
