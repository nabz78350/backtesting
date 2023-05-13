import os
import sys
import shutil
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import random
from scipy.stats import norm
import pandas_ta as ta
from pandas.tseries.offsets import BDay
import multiprocessing
from multiprocessing import Pool

def export_signal(signal,_snpdate):
    if signal.shape[0]==0:
        return
    
    pq_path = 'data/US/STRAT_EARNINGS/'
    if os.path.isdir(pq_path):
        pq_dataset = pq.ParquetDataset(path_or_paths=pq_path,use_legacy_dataset=False)
        pq_files = list(filter(lambda k:"SnpDate"+ str(_snpdate) in k,pq_dataset.files))
        for f in pq_files:
            dir_path = os.path.dirname(f)
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)

    tb = pa.Table.from_pandas(signal)
    pq.write_to_dataset(tb,root_path= pq_path,partition_cols=['SnpDate'])


def predict_earnings(simulations_earnings,dates_earnings,i,debt_cash,P):

    quarter = simulations_earnings[i]
    prev_quarter = simulations_earnings[i-1]
    print(quarter,prev_quarter)
    dates_train = dates_earnings.loc[prev_quarter,'dates_earn_cumul']
    dates_signal = dates_earnings.loc[quarter,'dates_earn']
    tickers = debt_cash[debt_cash.index.isin(dates_train)][P].iloc[-1].dropna().index.tolist()
    SIGNAL = pd.DataFrame(index = dates_signal,columns = tickers)

    for ticker in tickers:
        dates = []
        for date in dates_signal:
            dates.append(date)
            try :
                sub_df = debt_cash[debt_cash.index.isin(dates_train+dates)].dropna(0,'all')
                sub_df = sub_df.rank(axis=1,pct=True,ascending=True).clip(0.01,0.99)
                sub_df_ticker = sub_df[ticker].rolling(20).rank(pct=True,ascending=True)
                pred = ta.ebsw(sub_df_ticker,length=20).dropna().loc[date]
                SIGNAL.loc[date,ticker]= pred
            except :
                pass
        
    SIGNAL = pd.DataFrame(SIGNAL.stack(),columns = ['SIGNAL'])
    SIGNAL.index.names = ['Date','Ticker']
    SIGNAL = SIGNAL.reset_index()
    SIGNAL['SnpDate'] = SIGNAL['Date'].dt.strftime('%Y%m%d').astype(int)
    snp_dates = SIGNAL['SnpDate'].unique().tolist()
    for _snpdate in  snp_dates:
        export_signal(SIGNAL[SIGNAL['SnpDate']==_snpdate],_snpdate)
    
    return 0


def import_earnings():
    earnings = pd.read_parquet('data/US/earnings.pq')
    earnings['Date'] = pd.to_datetime(earnings['reportDate'])
    earnings = earnings.droplevel(0).set_index('Date',append=True)
    earnings = earnings.reorder_levels(['Date','Ticker'])
    earnings = earnings[~earnings.index.duplicated(keep='first')]
    earnings = earnings[earnings['epsActual'].notna()]
    earnings['EarningsQuarter'] = pd.to_datetime(earnings['reportDate']).dt.to_period('Q')
    earnings['IsReporting'] = 1
    return earnings

def import_cash_flows():
    cash_flow = pd.read_parquet('data/US/cash_flow.pq')
    cash_flow['Date'] = pd.to_datetime(cash_flow['filing_date'])
    cash_flow = cash_flow.droplevel(0).set_index('Date',append=True)
    cash_flow = cash_flow.reorder_levels(['Date','Ticker'])
    cash_flow = cash_flow[~cash_flow.index.duplicated(keep='first')]
    return cash_flow

def import_income_statements():
    income_statement = pd.read_parquet('data/US/income_statement.pq')
    income_statement['Date'] = pd.to_datetime(income_statement['filing_date'])
    income_statement = income_statement.droplevel(0).set_index('Date',append=True)
    income_statement = income_statement.reorder_levels(['Date','Ticker'])
    income_statement = income_statement[~income_statement.index.duplicated(keep='first')]
    return income_statement

def import_balance_sheets():
    balance_sheet = pd.read_parquet('data/US/balance_sheets.pq')
    balance_sheet['Date'] = pd.to_datetime(balance_sheet['filing_date'])
    balance_sheet = balance_sheet.droplevel(0).set_index('Date',append=True)
    balance_sheet = balance_sheet.reorder_levels(['Date','Ticker'])
    balance_sheet = balance_sheet[~balance_sheet.index.duplicated(keep='first')]
    return balance_sheet


def business_days(start_date,end_date):
    dates_list = pd.date_range(start=start_date,end=end_date,freq='D')
    bd_list = [date for date in dates_list if date.weekday()< 5]
    return bd_list


def DATES_EARNINGS(data:pd.DataFrame,start:float,end:float,_days_before:int,_days_after:int):

    all_earnings =data['EarningsQuarter'].dropna().unique().tolist()
    dates_earnings = pd.DataFrame(index = all_earnings,columns = ['start_pre','end_pre','start','end','start_post','end_post'])

    for earning in all_earnings:

        sample = data[data['EarningsQuarter']==earning][['IsReporting']]
        
        sample['IsReporting_%'] = sample['IsReporting'].div(sample['IsReporting'].sum())
        df = pd.DataFrame(sample.groupby(level=0)['IsReporting_%'].sum())
        tmp = (df['IsReporting_%']>start ) & (df['IsReporting_%']<end )

        try :
            dates_earnings.loc[earning,'start'] = df[tmp].index[0]
            dates_earnings.loc[earning,'end'] = df[tmp].index[-1]
            dates_earnings.loc[earning,'start_pre'] = dates_earnings.loc[earning,'start'] + BDay(-1- _days_before)
            dates_earnings.loc[earning,'end_pre'] = dates_earnings.loc[earning,'start'] + BDay(-1)
            dates_earnings.loc[earning,'start_post'] = dates_earnings.loc[earning,'end'] + BDay(1)
            dates_earnings.loc[earning,'end_post'] = dates_earnings.loc[earning,'end'] + BDay(1+_days_after)
            
        except :
            ''
    dates_earnings = dates_earnings.dropna()
    dates_earnings['dates_pre'] =   dates_earnings.apply(lambda x: business_days(x['start_pre'],x['end_pre']),axis=1)
    dates_earnings['dates_earn'] =   dates_earnings.apply(lambda x: business_days(x['start'],x['end']),axis=1)
    dates_earnings['dates_post'] =   dates_earnings.apply(lambda x: business_days(x['start_post'],x['end_post']),axis=1)
    dates_earnings['dates_earn_cumul'] = dates_earnings['dates_earn'].cumsum()

    return dates_earnings.sort_index()



def run_signal():
    P = pd.read_parquet('data/US/universe_table.pq')
    earnings = import_earnings()
    balance_sheet = import_balance_sheets()
    net_debt = balance_sheet['netDebt'].unstack().reindex_like(P).ffill().astype(float)
    cash = balance_sheet['cash'].unstack().reindex_like(P).ffill().astype(float)

    debt_cash = (net_debt/ cash)[P]
    dates_earnings = DATES_EARNINGS(earnings,0.03,0.97,10,10)
    simulations_earnings = dates_earnings.loc['2009Q4':'2023Q1'].index
    size = len(simulations_earnings)
    tuples = [(simulations_earnings,dates_earnings,j,debt_cash,P) for j in range(1,size)]
    multiprocessing.freeze_support()
    pool = Pool(processes=8)
    _ = pool.starmap(predict_earnings,iterable=tuples)
    pool.close()
    pool.join()

if __name__=='__main__':
    run_signal()