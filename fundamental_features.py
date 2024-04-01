import pandas as pd
import numpy as np
from data_master import DataMaster
from scipy.stats import norm
master = DataMaster()
from quantstats.stats import sharpe
import matplotlib.pyplot as plt
from tqdm import tqdm


def format_dataframe(df:pd.DataFrame,column: str):
    df.index = pd.to_datetime(df.index)  
    df = pd.DataFrame(df.stack())
    df.index.names = ['Date','Ticker']
    df.columns = column
    return df  
        
    
def get_chatoia(income_statement,balance_sheet,P):
    sell_change  =income_statement['operatingIncome'].unstack().reindex_like(P).ffill().astype(float).diff(252)
    total_assets = balance_sheet['totalAssets'].unstack().reindex_like(P).ffill().astype(float)
    chatoia = sell_change / total_assets
    chatoia = format_dataframe(chatoia, column = ['CHATOIA'])
    return chatoia


def get_mve(market_data,P):
    mkt_cap = market_data['MktCap'].unstack().reindex_like(P).ffill().astype(float)
    mve = np.log(mkt_cap)
    mve = format_dataframe(mve, column = ['MVE'])
    return mve

def get_sue(earnings,P):
    sue = earnings['epsDifference'].unstack().reindex_like(P).ffill()
    sue = format_dataframe(sue, column = ['SUE'])
    return sue

def get_turn(market_data,balance_sheet,P):
    volume = market_data['volume'].unstack().reindex_like(P).ffill()
    volume = volume.resample('M').mean().shift().reindex_like(P).ffill().astype(float)
    shares = balance_sheet['commonStockSharesOutstanding'].unstack().reindex_like(P).ffill().astype(float)
    turn = (volume/ shares).replace(np.inf, np.nan).clip(0,1)

def get_indmom(R,GICS,P):
    GICS_ = GICS.copy()
    R_sect = pd.DataFrame().reindex_like(R)
    results = {}
    for sect in tqdm(GICS_['gicsect'].unique().tolist()):
        tickers = GICS_[GICS_['gicsect']==sect].index.tolist()
        R_center = R[P].sub(R[P].mean(1),0)
        results[sect] =  R_center[tickers].sum(1).rolling(252).sum()
        for ticker in tickers:
            R_sect[ticker] = results[sect]

    R_sect = format_dataframe(R_sect, column = ['RSECT'])
    return R_sect

def get_grcapx(cash_flow,P):
    grcaps = cash_flow['capitalExpenditures'].unstack().reindex_like(P).ffill().astype(float)
    grcaps = grcaps.pct_change(506).clip(-3,3)
    grcaps = format_dataframe(grcaps, column = ['GRCAPS'])
    return grcaps


def get_ill(R,market_data,P):
    ill = R.abs()
    volume = market_data['volume'].unstack().reindex_like(P).ffill().astype(float)
    ill = ill / volume
    ill = format_dataframe(ill, column = ['ILL'])
    return ill

def get_ibq(earnings,P):
    eps = earnings['epsActual']
    eps_shift = eps.groupby(level=1).shift(4)
    # Identify quarters with an increase in earnings over the same quarter in the prior year
    eps_diff = ((eps-eps_shift) / eps_shift )>0.1 *1
    # Calculate the cumulative count of 1s, resetting the count to 0 when a 0 is encountered
    ibq = eps_diff.groupby(level=1).apply(
    lambda x: x.groupby(x.eq(0).cumsum()).cumcount() + x 
    ).droplevel(0)
    ibq = ibq.unstack().reindex_like(P).ffill()
    ibq = format_dataframe(ibq, column = ['IBQ'])
    return ibq

def get_ear(R,earnings,P):
    earnings_date = pd.to_datetime(earnings['reportDate'])
    earnings_date= earnings_date.unstack().reindex_like(P).ffill()
    earnings_date = earnings_date.sub(earnings_date.index,0)
    n_days =  - earnings_date.stack().dt.days.unstack()
    ear = R[n_days.abs()<=3].cumsum().reindex_like(P).ffill()
    ear = format_dataframe(ear, column = ['EAR'])
    return ear 

def get_retvol(R,P):
    retvol = R.resample('M').std().shift()
    retvol = retvol.reindex_like(P).ffill()
    retvol = format_dataframe(retvol, column = ['RETVOL'])
    return retvol

def get_cash(balance_sheet,P):
    cash = balance_sheet['cashAndEquivalents'].unstack().reindex_like(P).ffill().astype(float)
    total_assets = balance_sheet['totalAssets'].unstack().reindex_like(P).ffill().astype(float)
    cash = cash / total_assets
    cash = format_dataframe(cash, column = ['CASH'])
    return cash 

def get_agr(balance_sheet,P):
    total_assets = balance_sheet['totalAssets'].unstack().reindex_like(P).ffill().astype(float)
    agr = total_assets.pct_change(252).clip(-2,2)
    agr = format_dataframe(agr, column = ['AGR'])
    return agr 

def get_chsho(balance_sheet,P):
    shares_df = balance_sheet['commonStockSharesOutstanding'].unstack().reindex_like(P).ffill().astype(float)
    chcsho = shares_df.pct_change(252).clip(-2,2)
    chcsho = format_dataframe(chcsho, column = ['CHCSHO'])
    return chcsho

def get_chinv(balance_sheet,P):
    inv = balance_sheet['inventory'].unstack().reindex_like(P).ffill().astype(float).pct_change(252)
    total_assets = balance_sheet['totalAssets'].unstack().reindex_like(P).ffill().astype(float)
    chinv = inv/ total_assets
    chinv = format_dataframe(chinv, column = ['CHINV'])
    return chinv 

def get_chtx(income_statement,P):
    tax_exp = income_statement['incomeTaxExpense'].astype(float).groupby(level=1).pct_change(4)
    tax_exp = tax_exp.unstack().reindex_like(P).ffill()
    tax_exp = format_dataframe(tax_exp, column = ['TAX_EXP'])
    return tax_exp 

def get_invest(balance_sheet,P):
    total_assets = balance_sheet['totalAssets'].unstack().reindex_like(P).ffill().astype(float)
    gross_plant = balance_sheet['propertyPlantAndEquipmentGross'].astype(float).groupby(level=1).pct_change(4).clip(-1,1)
    inventory =  balance_sheet['inventory'].astype(float).groupby(level=1).pct_change(4).clip(-1,1)
    gross_plant = gross_plant.unstack().reindex_like(P).ffill().astype(float)
    inventory = inventory.unstack().reindex_like(P).ffill().astype(float)
    inv = (gross_plant + inventory)
    inv = inv / total_assets
    inv = format_dataframe(inv, column = ['INV'])
    return inv 

def get_currat(balance_sheet,P):
    total_assets = balance_sheet['totalCurrentAssets'].unstack().reindex_like(P).ffill().astype(float)
    total_liab = balance_sheet['totalCurrentLiabilities'].unstack().reindex_like(P).ffill().astype(float)
    currat = (total_assets / total_liab) 
    currat = format_dataframe(currat, column = ['CURRAT'])
    return currat 

def get_pchcurrat(balance_sheet,P):
    total_assets = balance_sheet['totalCurrentAssets'].unstack().reindex_like(P).ffill().astype(float)
    total_liab = balance_sheet['totalCurrentLiabilities'].unstack().reindex_like(P).ffill().astype(float)
    currat = (total_assets / total_liab) 
    pchcurrat = currat.pct_change(252)
    pchcurrat = format_dataframe(pchcurrat, column = ['PCHCURRAT'])
    return pchcurrat


    
def aggregate_fundamental_data():
    market_data = pd.read_parquet('data/US/mkt_data.pq')
    balance_sheet = pd.read_parquet('data/US/balance_sheets.pq')
    earnings = pd.read_parquet('data/US/earnings.pq')
    income_statement = pd.read_parquet('data/US/income_statement.pq')
    cash_flow = pd.read_parquet('data/US/cash_flow.pq')
    GICS = pd.read_parquet('data/US/GICS.pq')
    P = pd.read_parquet('data/US/universe_table.pq')
    R = market_data['close'].unstack().reindex_like(P).pct_change()
    
    fund_data = []
    fund_data.append(get_chatoia(income_statement,balance_sheet,P))
    fund_data.append(get_mve(market_data,P))
    fund_data.append(get_sue(earnings,P))
    fund_data.append(get_turn(market_data,balance_sheet,P))
    fund_data.append(get_indmom(R,GICS,P))
    fund_data.append(get_grcapx(cash_flow,P))
    fund_data.append(get_ill(R,market_data,P))
    fund_data.append(get_ibq(earnings,P))
    fund_data.append(get_ear(R,earnings,P))
    fund_data.append(get_retvol(R,P))
    fund_data.append(get_cash(balance_sheet,P))
    fund_data.append(get_agr(balance_sheet,P))
    fund_data.append(get_chsho(balance_sheet,P))
    fund_data.append(get_chinv(balance_sheet,P))
    fund_data.append(get_chtx(income_statement,P))
    fund_data.append(get_invest(balance_sheet,P))
    fund_data.append(get_currat(balance_sheet,P))

    fund_data = pd.concat(fund_data,axis=1).sort_index(level=1)
    return fund_data 



if __name__ == '__main__':
    fundamental_data = aggregate_fundamental_data()
    fundamental_data.to_parquet('data/US/fundamental_data.pq')
    