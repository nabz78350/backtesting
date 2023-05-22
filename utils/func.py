import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the system path
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
import datetime
import tqdm
from data_master import DataMaster
from tqdm import tqdm
from scipy.stats import norm
master = DataMaster()

def data_from_dict(dico:dict):

    data = pd.DataFrame()
    for key, value in tqdm(dico.items()):
        try :
            df = value
            df['Ticker'] = key
            data = pd.concat([df, data], 0)
        except :
            ''
    return data.set_index(['datetime','Ticker'])

def aggregate_tickers_classifs(tickers):
    classif = {}
    for ticker in tqdm(tickers) :
        try :
            classif_ticker = master.equities.get_ticker_classification(ticker,'US')
            classif[ticker] = classif_ticker
        except :
            classif = classif
    return pd.DataFrame(classif).T

def aggregate_tickers_balance_sheet(tickers):
    balance_sheet = {}
    for ticker in tqdm(tickers) :
        try :
            balance_sheet_ticker = master.equities.get_ticker_balance_sheet(ticker,'US','q')
            balance_sheet[ticker] = balance_sheet_ticker
        except :
            balance_sheet = balance_sheet
    balance_sheet = pd.concat(balance_sheet)
    balance_sheet.index.names =['Ticker','Date']
    balance_sheet = balance_sheet .reorder_levels(['Date','Ticker'])
    return balance_sheet

def aggregate_tickers_earnings(tickers):
    earnings = {}
    for ticker in tqdm(tickers) :
        try :
            earnings_ticker = master.equities.get_ticker_earnings_history(ticker,'US')
            earnings[ticker] = earnings_ticker
        except :
            earnings = earnings
    earnings = pd.concat(earnings)
    earnings.index.names =['Ticker','Date']
    earnings = earnings .reorder_levels(['Date','Ticker'])

    return earnings

def aggregate_tickers_income_statement(tickers):
    income_statement = {}
    for ticker in tqdm(tickers) :
        try :
            income_statement_ticker = master.equities.get_ticker_income_statement(ticker,'US','q')
            income_statement[ticker] = income_statement_ticker
        except :
            income_statement = income_statement
    income_statement = pd.concat(income_statement)
    income_statement.index.names =['Ticker','Date']
    income_statement = income_statement .reorder_levels(['Date','Ticker'])
    return income_statement

def aggregate_market_data(tickers:list,period_start :datetime.date):
    mkt_data = {}
    for ticker in tqdm(tickers) :
        try :
            mkt_data_ticker = master.equities.get_ohlcv(ticker,'US',period_start = period_start)
            mkt_cap =master.equities.get_ticker_historical_mcap('AAPL','US')
            mkt_data_ticker.index = mkt_data_ticker['datetime']
            mkt_data_ticker.index.names =['Date']
            mkt_cap.index.names =['Date']
            mkt_cap.columns =['MktCap']
            mkt_cap.index = pd.to_datetime(mkt_cap.index)
            mkt_data_ticker = mkt_data_ticker.join(mkt_cap,how ='left')
            mkt_data[ticker]= mkt_data_ticker
        except :
            mkt_data = mkt_data    
    mkt_data = pd.concat(mkt_data)
    mkt_data.index.names =['Ticker','Date']
    mkt_data = mkt_data .reorder_levels(['Date','Ticker'])
    return mkt_data



def aggregate_tickers_cash_flow(tickers):
    cash_flow = {}
    for ticker in tqdm(tickers) :
        try :
            cash_flow_ticker = master.equities.get_ticker_cash_flow(ticker,'US','q')
            cash_flow[ticker] = cash_flow_ticker
        except :
            cash_flow = cash_flow
    cash_flow = pd.concat(cash_flow)
    cash_flow.index.names =['Ticker','Date']
    cash_flow = cash_flow .reorder_levels(['Date','Ticker'])

    return cash_flow

def aggregate_tickers_dividends(tickers):
    dividends = {}
    for ticker in tqdm(tickers) :
        try :
            dividends_ticker = master.equities.get_historical_dividends(ticker,'US','q')
            dividends[ticker] = dividends_ticker
        except :
            dividends = dividends
    dividends = pd.concat(dividends)
    dividends.index.names =['Ticker','Date']
    dividends = dividends .reorder_levels(['Date','Ticker'])

    return dividends

def create_rank_column(df:pd.DataFrame,column :str,pct=True, ascending=True,level=0,normalize = False):
    column_rank = column+'_rank'
    if normalize:
        df[column_rank] = df.groupby(level=level)[column].rank(pct=pct,ascending=ascending).clip(0.01,0.99).apply(norm.ppf)
    else :
        df[column_rank] = df.groupby(level=level)[column].rank(pct=pct,ascending=ascending).clip(0.01,0.99)

    return df

def write_to_parquet(df:pd.DataFrame,directory:str,name :str):
    path_directory = 'data/'+directory
    path = path_directory +'/'+name +'.pq'
    # Check whether the specified path exists or not
    isExist = os.path.exists(path_directory)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path_directory)

    if os.path.exists(path):
        os.remove(path)
    
    df.to_parquet(path)
    print('data to parquet done -->',path)


def center(x):
    mean = x.mean(1)
    x = x.sub(mean,0)
    return x

if __name__=='__main__':
    tickers = ['AAPL']
    aapl_mkt = aggregate_market_data(tickers,datetime.date(2010,1,1))
    aapl_balance_sheet = aggregate_tickers_balance_sheet(tickers)
    print(aapl_mkt.dropna()) ### market cap is gave weekly, good to know to adjust
    print(aapl_balance_sheet)

