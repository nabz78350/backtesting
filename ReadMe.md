# Backtesting - L/S SP500 equities

This a short repo that contains some backtests on simple strategies.
The universe of investment is the SP500 historical composition since 2008.



## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Strategies](#strategies)
- [Improvement](#improvement)

## Project Overview

These signals are very simple, intuitive, but also noisy and definetely need improvement. They use simple mean reversion properties of US Stocks, or well-known (but also crowded) stylized facts on fundamental data.

## Data sources

Most of this work is data dependant. I got my data from eodhistorical data (https://eodhistoricaldata.com/). It's a pretty cheap datasource but quite reliable. I have tested other free options and that was the most easy one, especially to developp the library to extract historical data.

Still, this data is not from very high standard quality, and I'm sure some of these signals could turn out to be pure noise when tested on high quality data. But we do with what we have

The historical composition of our investment universe (SP500) was extracted from wikipedia webscrapping. 
(see 'utils/universe_selction.py')



## Data Extraction

The data is ectracted trhough eodhistorical data using several '.py' files

Some example data is extracted are porvided in the 'data_scrapping.ipynb' file

for instance for tickers AAPL and MSFT

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

All the extraction of data from eod api has been mostly inspired from diverse python found on the net and enhanced to serve the backtesting needs.



## Folder Structure

