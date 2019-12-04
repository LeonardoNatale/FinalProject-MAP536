import scipy.stats
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor, RandomForestRegressor
from Service.data_manager import DataManager
from Service.ramp_external_data_generator import RampExternalDataGenerator
from Model.ramp_model import RampModel
from sklearn.metrics import mean_squared_error

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
            "l2_regularization": scipy.stats.uniform(0, 1),
            "max_iter": scipy.stats.randint(low=800, high=1200),
            "max_depth": [5, 8, 10],
            "min_samples_leaf": [10, 15, 20, 25, 30]
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
    # x.fit(dm.get_train_X(), dm.get_train_y())
    # x.rmse(dm.get_test_X(), dm.get_test_y())
    # x.feature_importance(dm.get_test_X())
    # x.model_quality_testing(
    #     train_x=dm.get_train_X(),
    #     train_y=dm.get_train_y(),
    #     test_x=dm.get_test_X(),
    #     test_y=dm.get_test_y()
    # )

# if model:
#
#     model_type = HistGradientBoostingRegressor
#     fixed = {"loss": 'least_squares', "max_depth": 15}
#     opt = {
#         "RandomSearch": {
#             "l2_regularization": scipy.stats.uniform(0.001, 2),
#             "min_samples_leaf": scipy.stats.randint(low=20, high=50),
#             "max_iter": scipy.stats.randint(low=800, high=1000)
#         }
#     }
#     fixed = {"loss": 'least_squares'}
#     opt = {
#         "RandomSearch": {
#             "max_depth": [5, 10, 15, 20],
#             "min_samples_leaf": scipy.stats.randint(20, 100),
#             "l2_regularization": [0.0015, 0.0025, 0.0035, 0.0045, 0.0055, 0.0065, 0.0075, 0.0085, 0.0095],
#             "max_iter": [200, 500, 800, 1000, 1500, 2000]
#         }
#     }
#
#     x = RampModel(model_type, fixed_parameters=fixed, optimizable_parameters=opt)
#     dm = x.get_data_manager()
#     x.model_quality_testing(
#         train_x=dm.get_train_X(),
#         train_y=dm.get_train_y(),
#         test_x=dm.get_test_X(),
#         test_y=dm.get_test_y()
#     )

    # model_type = HistGradientBoostingRegressor
    # fixed = {"loss": 'least_squares',
    #          "learning_rate": 0.05,
    #          "max_iter": 1000,
    #          "max_leaf_nodes": 31,
    #          "min_samples_leaf": 20,
    #          "max_bins": 255,
    #          "scoring": mean_squared_error,
    #          "validation_fraction": 0.1,
    #          }
    #
    # opt = {
    #     "RandomSearch": {
    #         "l2_regularization": scipy.stats.uniform(0, 1),
    #         "max_depth": scipy.stats.randint(5, 10),
    #
    #     }
    # }
    # fixed = {"loss": 'least_squares'}
    # opt = {
    #     "RandomSearch": {
    #         "max_depth": [5, 10, 15, 20],
    #         "min_samples_leaf": scipy.stats.randint(20, 100),
    #         "l2_regularization": [0.0015, 0.0025, 0.0035, 0.0045, 0.0055, 0.0065, 0.0075, 0.0085, 0.0095],
    #         "max_iter": [200, 500, 800, 1000, 1500, 2000]
    #     }
    # }


# if model:
#     model_type = AdaBoostRegressor
#     fixed = {
#         "base_estimator": HistGradientBoostingRegressor(
#             l2_regularization=1.75,
#             max_depth=20,
#             max_iter=860,
#             min_samples_leaf=25
#         )
#     }
#     opt = {}
#     x = RampModel(model_type, fixed_parameters=fixed, optimizable_parameters=opt)
#     dm = x.get_data_manager()
#     x.model_quality_testing(
#         train_x=dm.get_train_X(),
#         train_y=dm.get_train_y(),
#         test_x=dm.get_test_X(),
#         test_y=dm.get_test_y()
#     )

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

