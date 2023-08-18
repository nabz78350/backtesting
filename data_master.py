import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to the system path
sys.path.append(parent_dir)

import json
import datetime
import asyncio
import pandas as pd
from fredapi import Fred
from eod import EodHistoricalData

import db_logs
import securities.misc as misc
import securities.baskets as baskets
import securities.equities as equities
import securities.crypto  as crypto
class DataMaster:

    def __init__(self, config_file_path="config.json"):
        with open(config_file_path, "r") as f:
            config = json.load(f)
            os.environ['EOD_KEY'] = config["EOD_KEY"]
            os.environ['FRED_KEY'] = config["FRED_KEY"]
        self.eod_client = EodHistoricalData(os.getenv('EOD_KEY'))
        self.fred_client = Fred(api_key=os.getenv("FRED_KEY"))
        self.oanda_client = "None"
        self.data_clients = {
            "eod_client": self.eod_client,
            "fred_client": self.fred_client,
            "oanda_client": self.oanda_client
        }

        self.misc = misc.Miscellaneous(data_clients=self.data_clients)
        self.baskets = baskets.Baskets(data_clients=self.data_clients)
        self.equities = equities.Equities(data_clients=self.data_clients)
        self.crypto = crypto.Crypto(data_clients=self.data_clients)
    
    def get_equity_service(self):
        return self.equities
    
    def get_misc_service(self):
        return self.misc
    
    def get_basket_service(self):
        return self.baskets
    


