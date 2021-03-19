# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
import requests
from requests import Session
from io import StringIO

import os
import pandas as pd
import datetime as dt
import logging

from mipi import gasgetter
from data_helper import test_internet_connection


def get_electricity_actuals(start, end, elexon_api_key, proxies):

    dfs = []
    day = start

    while day <= end:
        d = day.strftime("%Y-%m-%d")
        url = f"https://api.bmreports.com/BMRS/FUELHH/v1?APIKey={elexon_api_key}&SettlementDate={d}&Period=*&ServiceType=csv"

        r = requests.get(url, proxies=proxies)
        data = r.content.decode()

        try:
            df = pd.read_csv(
                StringIO(data), skiprows=1, skipfooter=1, header=None, engine="python",
            )
            df.columns = [
                "RECORDTYPE",
                "ELEC_DAY",
                "SETTLEMENT_PERIOD",
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
            df["ELEC_DAY"] = pd.to_datetime(df["ELEC_DAY"], format="%Y%m%d")

            if not df.empty:
                dfs.append(df)
        except:
            print("WARNING ", f"Elexon {d} not available from {url}")

        finally:
            day += dt.timedelta(1)
    if len(dfs) > 1:
        final_df = pd.concat(dfs)
    elif len(dfs) == 1:
        final_df = dfs[0]
    else:
        print("WARNING ", f"Elexon data not available for specified range")
    return final_df


def save_file(df, name, directory):
    df["CREATED_ON"] = dt.datetime.today()
    df.columns = df.columns.str.upper()

    df.to_csv(os.path.join(directory, f"{name}.csv"))


def main(start, end, output_dirpath, proxies):
    """ Runs data scripts to populate raw data in (../raw) from the internet.
    """
    logger = logging.getLogger(__name__)

    logger.info("testing internet connection")
    test_internet_connection("https://www.google.com", proxies)

    start = dt.datetime.strptime(start, "%Y-%m-%d")
    end = dt.datetime.strptime(end, "%Y-%m-%d")

    logger.info("grabbing electricity actuals")
    df = get_electricity_actuals(start, end, os.environ.get("elexon_api_key"), proxies)

    save_file(df, "ELECTRICITY_ACTUALS", output_dirpath)

    logger.info("connecting to gas data")
    gassy = gasgetter(
        shopping_list_path=r"{}".format(os.environ.get("shopping_list_path")),
        LatestFlag="Y",
        proxies=proxies,
    )
    logger.info("connected to gas data")

    mapping = {
        "GAS_VOLUME": "NTS Physical Flows",
        "GAS_ENERGY": "NTS Energy Offtaken",
        "GAS_CV": "Calorific Value",
        "GAS_GQ": "NTS SG Gas Quality",
        "temperature": "Temperature, Actual",
    }

    for key in mapping.keys():
        logger.info(f"grabbing {key}")
        try:
            df = gassy.get_all_filtered(start, end, filter=mapping[key])
            print(df.head())
            save_file(df, key, output_dirpath)
        except Exception as E:
            print(E)
            logging.critical(f"Error gathering {key}")

    logger.info("making final data set from raw data")


if __name__ == "__main__":
    print("lezzdoit")
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    print(project_dir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    dotenv_path = os.path.join(project_dir, ".env")
    load_dotenv(dotenv_path)
    # load_dotenv(find_dotenv())

    proxies = os.environ.get("proxies")
    if proxies != "None":
        proxies = {"https": proxies, "http": proxies}
    else:
        proxies = None

    main("2015-01-01", "2021-01-01", os.path.join(project_dir, "data/raw"), proxies)
