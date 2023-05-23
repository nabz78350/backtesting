import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import ssl
import requests
import urllib

ssl._create_default_https_context = ssl._create_unverified_context

def isoformat(date):
    date = pd.to_datetime(date)
    return date.isoformat()


wikipedia_pages = {'SPX': 'List of S&P 500 companies'}

def get_revisions_metadata(page: str, rvstart=None, rvend=None, rvdir: str = 'newer', rvlimit: int = 1, S=requests.Session(), **kwargs):
    """Get metadata for revision(s) using MediaWiki API
    Args:
        page: page title
        rvstart: get revisions starting from this date. Most common date formats are accepted. Default = None (no limit on rvstart)
        rvend: get revisions until this date. Most common date formats are accepted. Default = None (no limit on rvend)
        rvdir: direction of revision dates for results. 
            If 'newer', results are ordered old->new (rvstart < rvend). Convenient for getting the first revision after a given date.
            If 'older', results are ordered new->old (rvstart > rvend). Convenient for getting the latest revision before a given date.
            Default = 'newer' 
        S: HTTP session to use. Default = requests.Session()
        kwargs: additional params to pass to the MediaWiki API query. See https://www.mediawiki.org/wiki/API:Revisions
    Returns:
        Revision(s) metadata
    """
    api_url = "https://en.wikipedia.org/w/api.php"
    query_params = {
        "action": "query",
        "prop": "revisions",
        "titles": page,
        "rvprop": "ids|timestamp|user|comment",
        "rvslots": "main",
        "formatversion": "2",
        "format": "json",
        "rvlimit": rvlimit,
        "rvdir": rvdir,
    }
    # cleanup dates
    if rvstart is not None:
        query_params['rvstart'] = isoformat(rvstart)
    if rvend is not None:
        query_params['rvend'] = isoformat(rvend)
    # optional query_params
    for k, v in kwargs.items():
        query_params[k] = v
    r = S.get(url=api_url, params=query_params)
    data = r.json()
    pages = data["query"]["pages"]
    revisions = pages[0]['revisions']
    return revisions


def get_index_components_at(index: str = 'SPX', when: str = None) -> pd.DataFrame:
    """Returns index components at a given date, according to the latest update on Wikipedia before that date.
    Args:
        index: The index to get components for. Currently only 'SPX' is supported. Default = 'SPX'
        when: The date when to search components. Default = today
    Returns:
        
    """
    if when is None:
        when = datetime.datetime.today()
    page = wikipedia_pages[index]
    revisions = get_revisions_metadata(page, rvdir='older', rvstart=when) # get latest revision before 'when'
    revision = revisions[0]
    table = pd.read_html(f"https://en.wikipedia.org/w/index.php?title={urllib.parse.quote(page)}&oldid={revision['revid']}")
    #components_df = table[0]
    for df in table: # usually the components df will be table[0], but sometimes there is a table before that just holds comments about the article, which we ignore.
        if 'Symbol' in df.columns:
            data = df.set_index('Symbol')
            data.index.name ='Ticker'
            data['Presence']= None
            data['Presence']=True
            return data[['Presence']]
        elif 'Ticker' in df.columns:
            data = df.set_index('Ticker')
            data.index.name ='Ticker'
            data['Presence']=True
            return data[['Presence']]
        elif 'Ticker Symbol' in df.columns:
            data = df.set_index('Ticker Symbol')
            data.index.name ='Ticker'
            data['Presence']= None
            data['Presence']=True
            return data[['Presence']]
        elif 'Ticker symbol' in df.columns:
            data = df.set_index('Ticker symbol')
            data.index.name ='Ticker'
            data['Presence']= None
            data['Presence']=True
            return data[['Presence']]
        else :
            pass
    return None
        


def get_index_components_history(index: str = 'SPX', start_date=None, end_date=None, freq='M'):
    """Get the historical components between start_date and end_date at a given frequency (e.g. monthly)
    Args:
        index: The index to get components for. Default = 'SPX'
        start_date:
        end_date:
        freq: pandas frequency string (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases)
    Returns:
    """
    if end_date is None:
        end_date = datetime.date.today()
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    historical_components = {}
    for date in tqdm(dates):
        try :

            components_at_date = get_index_components_at(index=index, when=date)
            historical_components[str(date)] = (components_at_date)
        except :
            print(date)


    return historical_components



def presence_matrix(d:dict):
    # Convert keys to datetime objects
    d = {pd.to_datetime(k): v for k, v in d.items()}

    # Get all business days between the first and last key
    start_date = min(d.keys())
    # Use today's date as the end date
    end_date = pd.datetime.today()

    # If today is not a business day, use the previous business day as the end date
    if not end_date.weekday() in range(5):
        end_date = end_date - pd.tseries.offsets.BDay()
    all_business_days = pd.date_range(start_date, end_date, freq='B',closed= 'right')

    # Create an empty dataframe with columns equal to the set of all tickers in the dictionary
    tickers = list(set(ticker for tickers_list in d.values() for ticker in tickers_list))
    df = pd.DataFrame(columns=tickers, index=all_business_days)

    # Iterate through the keys of the dictionary and set the values of the corresponding tickers to True between the current key and the next key (if there is one)
    for i, key in enumerate(sorted(d.keys())):
        if i == len(d) - 1:
            # This is the last key in the dictionary, set all remaining values to False
            df.loc[key:, :] = False
        else:
            next_key = sorted(d.keys())[i+1]
            tickers_present = d[key]
            tickers_next = d[next_key]
            for ticker in tickers:
                if ticker in tickers_present:
                    df.loc[key:next_key-pd.Timedelta(days=1), ticker] = True
                elif ticker not in tickers_next:
                    df.loc[key:next_key-pd.Timedelta(days=1), ticker] = False

    # Fill any remaining NaN values with False
    df.fillna(False, inplace=True)

    return df