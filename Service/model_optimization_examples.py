import scipy.stats
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, BaggingRegressor, \
    AdaBoostRegressor, RandomForestRegressor
from Model.multi_model import MultiModel
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
    fixed = {}
    opt = {
        "RandomSearch": {
            "l2_regularization": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9, 1.2, 1.5, 1.7],
            "max_iter": [5000],
            "max_depth": [5],
            "min_samples_leaf": [10, 20, 30, 35, 40]
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

# ------------------ MULTI MODELS ------------------ #

if multi:
    multi = MultiModel(
        {
            "models": {
                GradientBoostingRegressor: {
                    "fixed_parameters": {
                        "loss": 'ls'
                    },
                    "optimizable_parameters": {
                        "RandomSearch": {
                            "alpha": scipy.stats.uniform(0, 1),
                            "max_depth": scipy.stats.uniform(2, 10)
                        },
                        "GridSearch": {
                            "criterion": ["friedman_mse", "mse"]
                        }
                    }
                }
            }
        }
    )

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
                ,
                GradientBoostingRegressor: {
                    "fixed_parameters": {
                        "loss": 'ls'
                    },
                    "optimizable_parameters": {
                        "RandomSearch": {
                            "alpha": scipy.stats.uniform(0, 1),
                            "max_depth": scipy.stats.uniform(2, 10)
                        },
                        "GridSearch": {
                            "criterion": ["friedman_mse", "mse"]
                        }
                    }
                }
            }
        }
    )

    print(multi.multi_model_testing())
