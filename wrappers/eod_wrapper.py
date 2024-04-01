import os
import json
import requests
import datetime
import calendar
import numpy as np
import pandas as pd

def get_fundamental_data(eod_client, ticker, exchange="US"):
    return eod_client.get_fundamental_equity("{}.{}".format(ticker, exchange))
       
def get_ohlcv(ticker="", exchange="US", period_end=datetime.datetime.today(), period_start=None, period_days=3650):
    url = "https://eodhistoricaldata.com/api/eod/{}.{}".format(ticker, exchange)
    params = {
        "api_token": os.getenv('EOD_KEY'), 
        "fmt": "json",
        "from": period_start.strftime('%Y-%m-%d') if period_start else (period_end - datetime.timedelta(days=period_days)).strftime('%Y-%m-%d'),
        "to": period_end.strftime('%Y-%m-%d')
    }
    resp = requests.get(url, params=params)
    df = pd.DataFrame(resp.json())
    df.rename(columns={"date": "datetime", "adjusted_close": "adj_close"}, inplace=True)
    if len(df) > 0:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def get_live_lagged_prices(ticker="", exchange="US"):
    #https://eodhistoricaldata.com/financial-apis/live-realtime-stocks-api/
    #supports multiple tickers pass
    url = "https://eodhistoricaldata.com/api/real-time/{}.{}".format(ticker, exchange)
    params = {
        "api_token": os.getenv('EOD_KEY'), 
        "fmt": "json"
    }
    resp = requests.get(url, params=params)
    return resp.json()

def get_intraday_data(ticker="", exchange="US", interval="60m", to_utc=datetime.datetime.utcnow(), period_days=120):

    url = "https://eodhistoricaldata.com/api/intraday/{}.{}".format(ticker, exchange)
    params = {
        "api_token": os.getenv('EOD_KEY'), 
        "interval": interval,
        "fmt": "json",
        "from": calendar.timegm((to_utc - datetime.timedelta(days=period_days)).utctimetuple()),
        "to": calendar.timegm(to_utc.utctimetuple())
    }
    resp = requests.get(url, params=params)
    df = pd.DataFrame(resp.json()).reset_index(drop=True).set_index("datetime")
    return df