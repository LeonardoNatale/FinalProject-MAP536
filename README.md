# FinalProject-MAP536

[More information here](https://ramp.r0h.eu/problems/air_passengers)

## How to optimize the hyperparameters of a model automatically?

If you want to optimize some hyperparameters of a model, follow these steps :

- Create a script, for instance `test.py`
- Add the following lines to your script.

``` python
from service.model import Model

model = Model(RandomForestRegressor) # Replace by your model name
x.optimize_model()
print(x.get_optimal_parameters())
```

- In order to make the code know which parameter to tune doing a RandomizedSearchCV, you need to add the following lines to `data/optimizable_parameters.json` :

``` json
"RandomForestRegressor" : { # Replace by your model
        "fixed_parameters" : { # These are the parameters for which you know the value
            "max_depth" : 50,
            "max_features" : 10
        },
        "optimizable_parameters" : { # These are the parameters you want to optimize by RandomizedSearchCV
            "randomforestregressor__n_estimators" : { # For each estimator, add the law to generate the values.
                "law" : "randint",
                "kwargs" : {
                    "low" : 10, 
                    "high" :100
                }
            },
            "randomforestregressor__max_depth" : {
                "law" : "randint",
                "kwargs" : {
                    "low" : 10, 
                    "high" :100
                }
            }
        }
    }
```
