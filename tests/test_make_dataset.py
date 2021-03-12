import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

import pandas as pd
import datetime as dt

from src.data.make_dataset import prepare_gas_volumes, prepare_electricity_actuals


def test_prepare_gas_volumes_basic(mocker):
    mock_data = pd.DataFrame(
        {
            "Unknown: 0": range(0, 5),
            "APPLICABLEAT": [dt.datetime(2020, 1, 1)] * 5,
            "APPLICABLEFOR": [dt.datetime(2020, 1, 1)] * 5,
            "VALUE": range(0, 5),
            "GENERATEDTIMESTAMP": ["2020-01-03 17:02:13"] * 5,
            "CREATEDDATE": ["2020-01-03 12:02:15"] * 5,
            "CREATED_ON": ["10/03/2021  13:30:53"] * 5,
            "ITEM": ["A", "B", "C", "D", "E"],
        }
    )

    def mock_get_data(self):
        return mock_data

    mocker.patch("src.data.make_dataset.get_data", mock_get_data)

    testdata = prepare_gas_volumes("testpath.csv")

    desired = pd.DataFrame(
        {
            "ITEM": ["A", "B", "C", "D", "E"],
            "GAS_DAY": [dt.datetime(2020, 1, 1)] * 5,
            "VALUE": range(0, 5),
        }
    )

    assert_frame_equal(desired, testdata)


def test_prepare_electricity_actuals_basic(mocker):

    mock_data = pd.DataFrame(
        {
            "SETTLEMENT_PERIOD": [1, 2, 3, 20, 21],
            "ELEC_DAY": [dt.datetime(2020, 1, 2)] * 5,
            "CREATED_ON": [dt.datetime(2020, 1, 2)] * 5,
        }
    )

    cols = [
        "CCGT",
        "OIL",
        "COAL",
        "NUCLEAR",
        "WIND",
        "PS",
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
    for col in cols:
        mock_data[col] = 10

    def mock_get_data(self):
        return mock_data

    mocker.patch("src.data.make_dataset.get_data", mock_get_data)

    testdata = prepare_electricity_actuals("testpath.csv")

    desired = mock_data.copy()
    desired["GAS_DAY"] = [
        dt.datetime(2020, 1, 1),
        dt.datetime(2020, 1, 1),
        dt.datetime(2020, 1, 1),
        dt.datetime(2020, 1, 2),
        dt.datetime(2020, 1, 2),
    ]

    assert_frame_equal(desired.sort_index(axis=1), testdata.sort_index(axis=1))
