# Backtesting - L/S SP500 equities

This a short repo that contains some backtests on simple strategies.
The universe of investment is the SP500 historical composition since 2008 (I couldn't find universe composition before that)

## Table of Contents
- [Project Overview](#project-overview)
- [Data Sources](#Data-sources)
- [Data Extraction](#Data-Extraction)
- [Utils function](#Utils-functions)

## Project Overview

These signals are very simple, intuitive, but also sometimes noisy and definitely need improvement. They use simple mean reversion or momentum properties of US Stocks, or well-known (but also crowded) stylized facts on fundamental data.

I've read some papers on earnings and tried to implement them, there is also some signals on well-known technical indicators or intuitive ideas on price/volume data

Some of these signals are non tradable as the bias is small and they have too much turnover. I will be working on trying to improve these signals with a more rigorous approach.


## Data sources

Most of this work is data dependant. I got my data from eodhistorical data (https://eodhistoricaldata.com/). It's a pretty cheap datasource but quite reliable (I got it for free from university). I have tested other free options and that was the most easy one, especially to developp the library to extract historical data.

Still, this data is not from very high standard quality, and I'm sure some of these signals could turn out to be pure noise when tested on high quality data. But we do with what we have. I've ran many checks on the data, and I have encountered small errors, missing data, but hopefully barely no data leaks (earnings, income statement or whatever fundamental data that is disclosed after the date stated in eodhistoricaldata)

The historical composition of our investment universe (SP500) was extracted from wikipedia webscrapping. 
(see 'utils/universe_selction.py')

Data is stored as parquet files under US/

- mkt_data : data for open,high,close,low,volume and mkt cap
- earnings : data for earning dates, eps and eps estimate
- dividends : data for dividend date and dividend value
- balance_sheet : data for debt, assets, (all kind of balance sheet features)
- income statement : data about net sales, cost of goods sold, margin, interest paid etc
- cash flow   : details of actual stocks cash flows : depreciation, sales, investments, wages, etc.
- presence_table : the P dataframe, historical composition of the SP500 (see utils/universe_selection.py)

## Data Extraction

The data is extracted through eodhistorical data using several '.py' files

Some example data is extracted are provided in the 'data_scrapping.ipynb' file

For instance for tickers AAPL and MSFT

```python
### get SP500 components between two dates
historical_components = get_index_components_history(start_date='2021-01-01',end_date='2021-01-10')
P = pd.concat(historical_components)
P = P['Presence'].unstack()
P.index = pd.to_datetime(P.index).date
write_to_parquet(P,'US','test_Presence')
```

```python 
### mkt data for AAPL and MSFT
tickers = P.columns.tolist()
from utils.func import aggregate_market_data
tickers = ['AAPL','MSFT']
mkt_data = aggregate_market_data(tickers,datetime.date(2008,1,1))
write_to_parquet(mkt_data,'US','test_mkt_data')
```

More examples can be found in 'data_scrapping.ipynb'. This script need to be run on all tickers of the Presence matrix (P dataframe)

The functions used to aggragte data are under 'utils.func.py'. Each of the aggregation function use extraction functions from the eodhistoricaldata api (see  'data_master.py', 'db_manager.py' and the securities folder)

All the function to extract the data from eod api has been mostly inspired from diverse python libraries found on the internet and that I enhanced to better serve my needs.


## Utils functions

Some stuff you need to know 

 - P in the backtests is the presence_matrix, a dataframe with True or False to state if the ticker is  in the SP500 for the specified datetime index

 That's why I often reindex other dataframes like P

```python
earnings = earnings['epsActual'].reindex_like(P)
```

P is also used as a filter to be sure to rank, or compare or trade stocks only when its in the investment universe

```python
signal = signal[P].rank(axis=1,ascending=True,pct=True)
```

The hedge of the signal are often based on stocks gics sector, group or industry 
(see center function in 'utils/py')
All these signals have constant nominal equal to 1 and no exposure (sum weights =0) in global and for the selected hedge. They may still be playing industry momentum or have beta by the idea behind the signal.

```python
signal = signal[P].groupby(GICS['gicsect'],axis=1).apply(center)
```
I'll try to write more documentation. All simulated pnls can be found in pnl_correlation.ipynb

If you see any non-sense, stupidities or anything wrong or supsicious, please reach !

Martin Boutier
martin.boutier@ensae.fr


