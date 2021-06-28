# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import datetime as dt

"""
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
"""


def prepare_gas_data(filepath, keep_powerstations_only=True):
    df = get_data(filepath)
    df = df.rename({"APPLICABLEFOR": "GAS_DAY"}, axis=1)
    df = df.drop_duplicates()
    df = keep_latest(df, cols=["GAS_DAY", "ITEM"])
    df = df.set_index("GAS_DAY").tz_localize(None).reset_index()
    df["POWERSTATION"] = df["ITEM"].str.contains("POWER STATION")

    if keep_powerstations_only:
        df = df[df["POWERSTATION"]]

    return df[["ITEM", "GAS_DAY", "VALUE"]].sort_values(["GAS_DAY", "ITEM"])


def prepare_electricity_actuals(filepath):
    elec = get_data(filepath)

    elec["ELEC_DATETIME"] = (
        elec["ELEC_DAY"] + dt.timedelta(minutes=30) * elec["SETTLEMENT_PERIOD"]
    )
    elec["GAS_DAY"] = pd.to_datetime(
        (elec["ELEC_DATETIME"] + dt.timedelta(hours=-5)).dt.date
    )

    elec = keep_latest(elec, cols=["ELEC_DATETIME"]).reset_index(drop=True)

    cols = [
        col
        for col in elec.columns
        if col
        not in [
            "CREATED_ON",
            "RUNID",
            "ELEC_DATETIME",
            "ELEC_DAY",
            "SETTLEMENT_PERIOD",
            "RECORDTYPE",
        ]
    ]
    elec = elec.groupby(["GAS_DAY"])[cols].mean()

    elec["TED"] = elec.sum(axis=1)

    return elec


def get_data(filepath):
    df = pd.read_csv(filepath)
    for col in ["APPLICABLEFOR", "APPLICABLEAT", "CREATED_ON", "ELEC_DAY"]:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass

    df.columns = df.columns.str.upper()

    return df


def keep_latest(df, cols=["GAS_DAY"]):
    df = df.sort_values(cols + ["CREATED_ON"])
    df = df.groupby(cols).last()
    return df.reset_index()


def map_to_sites(df):
    # grab the site from the item name
    mapping = {}
    for item in df["ITEM"].unique():
        try:
            site = item.split(",")[1].strip()
            mapping[item] = site
        except:
            print(item)
    df["SITE"] = df["ITEM"].map(mapping)
    df["SITE"] = df["SITE"].astype("category")
    return df


'''
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    elec = prepare_electricity_actuals(os.path.join(input_filepath, "ELECTRICITY_ACTUALS.csv")
    daily_elec_averages = elec[["CCGT", "OCGT"]].fillna(0).sum(axis=1)
    daily_elec_GWH=daily_elec_averages * 24 / 1000
    
    gas_energy=make_dataset.prepare_gas_data(os.path.join(input_filepath, "GAS_ENERGY.csv").rename({"VALUE": "ENERGY"}, axis=1)
    gas_energy=map_to_sites(gas_energy)
    gas_energy["ENERGY"] = gas_energy["ENERGY"].str.replace(",", "").astype(float)
    gas_energy["ENERGY_GWH"] = gas_energy["ENERGY"] / 1000000

    # calculate the daily average energy for all Powerstations
    daily_gas_energy = (
    gas_energy[gas_energy["POWERSTATION"]]
    .groupby("GAS_DAY")["ENERGY_GWH"]
    .sum()
    .tz_localize(None)
)

    df = pd.DataFrame({"ELECTRICITY": daily_elec_GWH, "GAS": daily_gas_energy}).dropna()
    df["EFFICIENCY"] = df["ELECTRICITY"] / df["GAS"]

    gas_volume=make_dataset.prepare_gas_data(os.path.join(input_filepath, "GAS_VOLUME.csv").rename({"VALUE": "VOLUME"}, axis=1)
    gas_volume=map_to_sites(gas_volume)

    gas_cv=make_dataset.prepare_gas_data(os.path.join(input_filepath, "GAS_CV.csv").rename({"VALUE": "CV"}, axis=1)
    gas_cv=map_to_sites(gas_cv)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
'''
