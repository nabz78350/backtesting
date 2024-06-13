import pandas as pd
import numpy as np
import datetime
import tqdm
import os
from tqdm import tqdm
from utils.universe_selection import *
import matplotlib.pyplot as plt
from utils.func import * 
P = pd.read_parquet('data/US/test_adv_table.pq')

def main():
    tickers = P.columns.tolist()
    
    mkt_data = aggregate_market_data(tickers,datetime.date(2000,1,1))
    write_to_parquet(mkt_data,'US','test_mkt_data')
    
    # balance_sheet = aggregate_tickers_balance_sheet(tickers)
    # write_to_parquet(balance_sheet,'US','test_balance_sheet')
    
    # GICS = aggregate_tickers_classifs(tickers)
    # write_to_parquet(GICS,'US','test_GICS')
    
    # earnings = aggregate_tickers_earnings(tickers)
    # write_to_parquet(earnings,'US','test_earnings')
    
    # cash = aggregate_tickers_cash_flow(tickers)
    # write_to_parquet(cash,'US','test_cash')
    
    # income_statement = aggregate_tickers_income_statement(tickers)
    # write_to_parquet(income_statement,'US','test_income_statement') 
    
    # dividends = aggregate_dividend_data(tickers = tickers)
    # write_to_parquet(dividends,'US','test_dividends') 
    

    return None


def main_intraday_data(period_days:int=1535,interval:str = "1h"):
    P_intraday = P.tail(period_days)
    P_intraday = P_intraday[P_intraday].dropna(axis=1,how ='all')
    tickers = P.columns.tolist()
        
    intraday_data= aggregate_intraday_data(tickers,period_days=period_days,interval=interval)
    write_to_parquet(intraday_data,'US','test_intraday_data_'+interval)
    

if __name__ =='__main__':
    main()
    # main_intraday_data()
