import requests
import logging


def test_internet_connection(url, proxies):
    try:
        r = requests.get(url, proxies=proxies)
        logging.debug(f"Connected successfully to {url}")
        return True

    except Exception as E:
        print(E)
        logging.critical(f"Connected failed to {url}")
        return False
