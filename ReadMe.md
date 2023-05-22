# Backtesting - L/S SP500 equities

This a short repo that contains some backtests on simple strategies.
The universe of investment is the SP500 historical composition since 2008.

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Strategies](#strategies)
- [Improvement](#improvement)

## Project Overview

These signals are very simple, intuitive, but also sometimes noisy and definitely need improvement. They use simple mean reversion, momentum properties of US Stocks, or well-known (but also crowded) stylized facts on fundamental data.

Some of these signals are non tradeable as the bias is small and they have too much turnover. I will be working on trying to improve these signals with a more rigorous approach.


## Data sources

Most of this work is data dependant. I got my data from eodhistorical data (https://eodhistoricaldata.com/). It's a pretty cheap datasource but quite reliable (I got it for free from university). I have tested other free options and that was the most easy one, especially to developp the library to extract historical data.

Still, this data is not from very high standard quality, and I'm sure some of these signals could turn out to be pure noise when tested on high quality data. But we do with what we have. I've ran many checks on the data, and I have encountered into small errors, missing data, but hopefully barely bo data leaks (earnings, income statement or whatever fundamental data that is disclosed after the date stated in eodhistoricaldata)

The historical composition of our investment universe (SP500) was extracted from wikipedia webscrapping. 
(see 'utils/universe_selction.py')

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


## Key variables used in the backtests

Some stuff you need to know 

 - P in the backtests is the presence_matrix, a dataframe with True or False to state if the ticker is  in the SP500 for the specified datetime index

 That's why I reindex other data like P

```python
earnings = earnings['epsActual'].reindex_like(P)
```

P is also used as a filter to be sure to rank, or compare or trad stocks only when its in the investment universe

```python
signal = signal[P].rank(axis=1,ascending=True,pct=True)
```


The hedge of the signal are often based on stocks gics sector, industry 
(see center function in 'utils/py')


```python
signal = signal[P].groupby(GICS['gicsect'],axis=1).apply(center)
```

If you see any non-sense, stupidities or anything wrong or supsicious, please reach !

Martin Boutier
martin.boutier@ensae.fr


