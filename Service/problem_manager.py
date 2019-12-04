class Problem:
    """
    This class is in charge of managing a Problem i.e. the configuration of the model implementation.
    """

    def __init__(self):
        self._problem_title = "Number of air passengers prediction"
        self._target_column_name = "log_PAX"
        self._train_f_name = "train.csv.bz2"
        self._test_f_name = "test.csv.bz2"
        self._ed_model_columns = [
            "Date",
            "AirPort",
            "Mean TemperatureC",
            "Mean Humidity",
            "Mean Sea Level PressurehPa",
            "Mean VisibilityKm",
            "Mean Wind SpeedKm/h",
            "CloudCover",
            "is_holiday",
            "is_beginning_holiday",
            "is_end_holiday",
            "holidays_distance",
            "gdp",
            "coordinates",
            "fuel_price",
            "passengers"
        ]
        self._ext_data_f_names = {
            "weather": "weather_data.csv",
            "airports": "airport-codes.csv",
            "gdp": "gdp.csv",
            "external_data": "external_data.csv",
            "jet_fuel": "jetfuel.csv",
            "passengers": "passengers.csv",
            "pax_aggregations": "aggregated_PAX.csv"
        }

    def get_target_column_name(self):
        return self._target_column_name

    def get_train_f_name(self):
        return self._train_f_name

    def get_test_f_name(self):
        return self._test_f_name

    def get_ed_model_columns(self):
        return self._ed_model_columns

    def get_ext_data_f_names(self):
        return self._ext_data_f_names
