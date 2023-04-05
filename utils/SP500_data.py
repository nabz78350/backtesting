from data_master import DataMaster
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
from data_master import DataMaster
from data_master import DataMaster
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
master = DataMaster()
from scipy.stats import norm
from utils.func import *
from utils import universe_selection



if __name__ =='__main__':
    SP500_presence = universe_selection.get_index_components_history('SPX','2008-01-01')
    P = universe_selection.presence_matrix(SP500_presence)
    tickers = P.columns.tolist()
    sectors = aggregate_tickers_classifs(tickers)
    write_to_parquet(sectors,'SP500','GICS')    

    mkt_data = aggregate_market_data(tickers,period_start=datetime.date(2008,1,1))
    write_to_parquet(sectors,'SP500','mkt_data')
    
      

