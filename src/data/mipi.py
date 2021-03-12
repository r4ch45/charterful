import requests
from requests import Session

from requests.adapters import HTTPAdapter, Retry
from zeep import Client
from zeep.helpers import serialize_object
from zeep.transports import Transport

import pandas as pd
import datetime as dt

import logging


class gasgetter:
    # http://marketinformation.natgrid.co.uk/
    # https://www.nationalgrid.com/uk/gas-transmission/sites/gas/files/documents/8589935564-API%20Guidance%20v1.0%2020.6.16.pdf

    def __init__(self, shopping_list_path, LatestFlag="Y", proxies=None):
        self.LatestFlag = LatestFlag
        self.proxies = proxies
        self.shopping_list_path = shopping_list_path
        self.shopping_list = pd.read_excel(shopping_list_path)

        # create session to accomodate proxies
        session = Session()
        session.proxies = self.proxies
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        url = "http://marketinformation.natgrid.co.uk/MIPIws-public/public/publicwebservice.asmx?wsdl"
        self.client = Client(url, transport=Transport(session=session))

    def get_all_filtered(self, start, end, filter):

        item_list = self.filter_items(filter)

        dfs = []
        for itm in item_list:
            df = self.get_individual_item_for_range(start, end, item=itm, max_range_days=366)

            if df is not None:
                df["ITEM"] = itm
                dfs.append(df)
            else:
                logging.warning(f"Issue gathering {itm}")

        if len(dfs) > 0:
            total_df = pd.concat(dfs)
            print(total_df.head())

        else:
            logging.critical(f"{filter} returned no data")
            total_df = None

        return total_df

    def filter_items(
        self, filter="NTS Power Station, NTS Physical Flows", ignore="METER"
    ):

        df = self.shopping_list
        df["Data Item"] = df["Data Item"].str.upper().copy()
        df = df[df["Data Item"].str.contains(str.upper(filter))]
        df = df[~df["Data Item"].str.contains(str.upper(ignore))]

        if len(df["Data Item"]) == 0:
            logging.critical(
                f"MIPI: Filter for {filter} returned no items, try rewording your filter"
            )
        return df["Data Item"].values

    def get_individual_item_for_range(self, start, end, item, max_range_days=365):
        dfs = []
        period_start = start
        period_end = min(start + dt.timedelta(days=max_range_days), end)
        logging.debug(f"MIPI: Gathering {item} - {period_start} to {end}")

        while period_end <= end:
            df = self.get_individual_item(period_start, min(period_end, end), item)
            dfs.append(df)
            period_start = period_end
            period_end = period_end + dt.timedelta(days=max_range_days)

        if (len(dfs) > 0) and all([d is not None for d in dfs]):
            total_df = pd.concat(dfs)
            return total_df
        else:
            logging.critical(f"MIPI: {item} returned nothing",)
            return None

    def get_individual_item(self, start, end, item):
        ApplicableForFlag = "Y"
        dateType = "GASDAY"

        fromDate = start.strftime("%Y-%m-%d")
        toDate = end.strftime("%Y-%m-%d")
        logging.debug(f"MIPI: Gathering {item} - {fromDate} to {toDate}",)

        body = {
            "LatestFlag": f"{self.LatestFlag}",
            "ApplicableForFlag": f"{ApplicableForFlag}",
            "ToDate": f"{toDate}",
            "FromDate": f"{fromDate}",
            "DateType": f"{dateType}",
            "PublicationObjectNameList": {"string": f"{item}"},
        }
        try:
            r = self.client.service.GetPublicationDataWM(body)
        except:
            logging.critical(
                f"MIPI: {item} request error, try checking if item exists",
            )
            r = None

        if r is not None:
            data = r[0].PublicationObjectData["CLSPublicationObjectDataBE"]
            data_dic = [serialize_object(d) for d in data]
            df = pd.DataFrame(data=data_dic, columns=data_dic[0].keys())
            logging.debug(f"MIPI: {item} {fromDate} to {toDate} gathering complete",)

            return df

        else:
            return None

