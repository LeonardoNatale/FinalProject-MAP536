import holidays
import datetime
import pandas as pd


class HolidaysManager:
    """
    This class is used to manage holidays.
    """

    def __init__(self, date):
        # Conversion to datetime
        if isinstance(date, datetime.datetime):
            self._date = date
        elif isinstance(date, str):
            self._date = datetime.datetime.strptime(date, '%Y-%m-%d')

    @staticmethod
    def to_holiday(series):
        """
        Converts every element of a pandas Series to a HolidayManager object.
        :param series: The pandas Series to convert.
        :return: The transformed Series.
        """
        return series.apply(lambda x: HolidaysManager(x))

    def __str__(self):
        return self._date.__str__()

    def is_holiday(self):
        """
        Returns True if the given day is a holiday in the US.
        """
        return self._date in holidays.US()

    def is_beginning_holiday(self):
        """
        Returns True is the day is a holiday and if the day before is not.
        """
        return self._is_holiday() and not self._date - datetime.timedelta(days=1) in holidays.US()

    def is_end_holiday(self):
        """
        Returns True is the day is a holiday and if the day after is not.
        """
        return self._is_holiday() and not self._date + datetime.timedelta(days=1) in holidays.US()
