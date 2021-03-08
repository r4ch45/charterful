import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

import pandas as pd

from src.data.create_raw_data import get_electricity_actuals

def test_get_electricity_actuals_basic(mocker):
    """
    Testing basic functionality with no errors
    """

    mock_data = ""

    def mock_get_data(self, tbl):
        return mock_data

    mocker.patch("requests.get", mock_get_data)

    actuals = get_electricity_actuals("2020-01-01", "2020-01-01")

    desired = pd.DataFrame({
        "ELEC_DAY" : pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "SETTLEMENT_PERIOD" : [1,2]})

    desired["RECORDTYPE"] = "FUELHH"
    columns = [
                        "CCGT",
                        "OIL",
                        "COAL",
                        "NUCLEAR",
                        "WIND",
                        "PUMPEDSTORAGE",
                        "NPSHYD",
                        "OCGT",
                        "OTHER",
                        "INTFR",
                        "INTIRL",
                        "INTNED",
                        "INTEW",
                        "BIOMASS",
                        "INTNEM",
                        "INTELEC",
                        "INTIFA2",
                        "INTNSL",
        ]

    for col in columns:
        desired[col] = 10

    assert_frame_equal(desired, demand)
