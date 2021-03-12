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


def prepare_gas_volumes(filepath):
    volume = get_data(filepath)
    volume = volume.rename({"APPLICABLEFOR": "GAS_DAY"}, axis=1)
    volume = volume.drop_duplicates()

    return volume[["ITEM", "GAS_DAY", "VALUE"]].sort_values(["GAS_DAY", "ITEM"])


def prepare_electricity_actuals(filepath):
    elec = get_data(filepath)

    elec["ELEC_DATETIME"] = (
        elec["ELEC_DAY"] + dt.timedelta(minutes=30) * elec["SETTLEMENT_PERIOD"]
    )
    elec["GAS_DAY"] = pd.to_datetime(
        (elec["ELEC_DATETIME"] + dt.timedelta(hours=-5)).dt.date
    )

    elec = keep_latest(elec, cols=["ELEC_DATETIME"]).reset_index()

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
    return df


'''
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


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
