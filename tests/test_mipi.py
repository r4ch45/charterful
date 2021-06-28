import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

import datetime as dt
import pandas as pd
import numpy as np
import os
import isodate

from src.data.mipi import gasgetter


# gassy = gasgetter(
#        shopping_list_path=r"{}".format(os.environ.get("shopping_list_path")),
#        LatestFlag="Y",
#        proxies=proxies,
#    )


def test_shopping_list_path():
    gassy = gasgetter(
        shopping_list_path=r"{}".format(os.environ.get("shopping_list_path")),
        LatestFlag="Y",
        proxies=None,
    )

    actual = list(gassy.shopping_list.columns)
    need = ["Data Dictionary Category", "Data Item"]

    assert set(need).issubset(actual)

    return


def test_get_all_filtered_basic():

    proxies = os.environ.get("proxies")
    if proxies != "None":
        proxies = {"https": proxies, "http": proxies}
    else:
        proxies = None

    gassy = gasgetter(
        shopping_list_path=r"{}".format(os.environ.get("shopping_list_path")),
        LatestFlag="Y",
        proxies=proxies,
    )

    df = gassy.get_all_filtered(
        dt.datetime(2020, 1, 1),
        dt.datetime(2020, 1, 2),
        filter = "NTS Energy Offtaken, BlackBridge, NTS Power Station",
    )

    desired = pd.DataFrame(
        {
            "ApplicableAt": pd.to_datetime(
                ["2020-01-01 00:00:00+00:00", "2020-01-02 00:00:00+00:00"],
            ).tz_convert(tz=isodate.tzinfo.UTC),
            "ApplicableFor": pd.to_datetime(
                ["2020-01-01 00:00:00+00:00", "2020-01-02 00:00:00+00:00"]
            ).tz_convert(tz=isodate.tzinfo.UTC),
            "Value": ["64244444", "63811111"],
            "GeneratedTimeStamp": pd.to_datetime(
                ["2020-01-04 10:41:12+00:00", "2020-01-03 12:02:15+00:00"]
            ).tz_convert(tz=isodate.tzinfo.UTC),
            "QualityIndicator": ["L", "L"],
            "Substituted": ["N", "N"],
            "CreatedDate": pd.to_datetime(
                ["2020-01-04 10:41:12+00:00", "2020-01-03 12:02:15+00:00"]
            ).tz_convert(tz=isodate.tzinfo.UTC),
            "ITEM": [
                "NTS ENERGY OFFTAKEN, BLACKBRIDGE, NTS POWER STATION",
                "NTS ENERGY OFFTAKEN, BLACKBRIDGE, NTS POWER STATION",
            ],
        },
        index=[0, 1],
    )
    desired["Value"] = desired["Value"].astype("object")

    assert_frame_equal(desired, df)

    return


def test_filter_items_basic():
    proxies = os.environ.get("proxies")
    if proxies != "None":
        proxies = {"https": proxies, "http": proxies}
    else:
        proxies = None

    gassy = gasgetter(
        shopping_list_path=r"{}".format(os.environ.get("shopping_list_path")),
        LatestFlag="Y",
        proxies=proxies,
    )

    filtered = gassy.filter_items(
        filter="NTS Energy Offtaken, BlackBridge, NTS Power Station"
    )

    desired = np.array(
        ["NTS ENERGY OFFTAKEN, BLACKBRIDGE, NTS POWER STATION"], dtype="object"
    )

    assert filtered == desired
    return


def test_get_individual_item_for_range_basic():

    proxies = os.environ.get("proxies")
    if proxies != "None":
        proxies = {"https": proxies, "http": proxies}
    else:
        proxies = None

    gassy = gasgetter(
        shopping_list_path=r"{}".format(os.environ.get("shopping_list_path")),
        LatestFlag="Y",
        proxies=proxies,
    )

    df = gassy.get_individual_item_for_range(
        dt.datetime(2020, 1, 1),
        dt.datetime(2020, 1, 2),
        item = "NTS Energy Offtaken, BlackBridge, NTS Power Station",
    )

    desired = pd.DataFrame(
        {
            "ApplicableAt": pd.to_datetime(
                ["2020-01-01 00:00:00+00:00", "2020-01-02 00:00:00+00:00"],
            ).tz_convert(tz=isodate.tzinfo.UTC),
            "ApplicableFor": pd.to_datetime(
                ["2020-01-01 00:00:00+00:00", "2020-01-02 00:00:00+00:00"]
            ).tz_convert(tz=isodate.tzinfo.UTC),
            "Value": ["64244444", "63811111"],
            "GeneratedTimeStamp": pd.to_datetime(
                ["2020-01-04 10:41:12+00:00", "2020-01-03 12:02:15+00:00"]
            ).tz_convert(tz=isodate.tzinfo.UTC),
            "QualityIndicator": ["L", "L"],
            "Substituted": ["N", "N"],
            "CreatedDate": pd.to_datetime(
                ["2020-01-04 10:41:12+00:00", "2020-01-03 12:02:15+00:00"]
            ).tz_convert(tz=isodate.tzinfo.UTC),
        },
        index=[0, 1],
    )
    desired["Value"] = desired["Value"].astype("object")

    assert_frame_equal(desired, df)

    return


def test_get_individual_item_basic():

    proxies = os.environ.get("proxies")
    if proxies != "None":
        proxies = {"https": proxies, "http": proxies}
    else:
        proxies = None

    gassy = gasgetter(
        shopping_list_path=r"{}".format(os.environ.get("shopping_list_path")),
        LatestFlag="Y",
        proxies=proxies,
    )

    df = gassy.get_individual_item(
        dt.datetime(2020, 1, 1),
        dt.datetime(2020, 1, 2),
        "NTS Energy Offtaken, BlackBridge, NTS Power Station",
    )

    desired = pd.DataFrame(
        {
            "ApplicableAt": pd.to_datetime(
                ["2020-01-01 00:00:00+00:00", "2020-01-02 00:00:00+00:00"],
            ).tz_convert(tz=isodate.tzinfo.UTC),
            "ApplicableFor": pd.to_datetime(
                ["2020-01-01 00:00:00+00:00", "2020-01-02 00:00:00+00:00"]
            ).tz_convert(tz=isodate.tzinfo.UTC),
            "Value": ["64244444", "63811111"],
            "GeneratedTimeStamp": pd.to_datetime(
                ["2020-01-04 10:41:12+00:00", "2020-01-03 12:02:15+00:00"]
            ).tz_convert(tz=isodate.tzinfo.UTC),
            "QualityIndicator": ["L", "L"],
            "Substituted": ["N", "N"],
            "CreatedDate": pd.to_datetime(
                ["2020-01-04 10:41:12+00:00", "2020-01-03 12:02:15+00:00"]
            ).tz_convert(tz=isodate.tzinfo.UTC),
        },
        index=[0, 1],
    )
    desired["Value"] = desired["Value"].astype("object")

    assert_frame_equal(desired, df)

    return

