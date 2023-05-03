import pandas as pd
import numpy as np
from data_master import DataMaster
from utils import func
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
master = DataMaster()
import pyarrow as pa
import pyarrow.parquet as pq
from utils.func import center
from quantstats.stats import sharpe
from sklearn.ensemble import GradientBoostingRegressor


class Step :
    
    def __init__(self,curr_date,X_train,Y_train):
        self.name_model = 'SVM'
        self.curr_date = curr_date
        self.snp_date = int(self.curr_date.strftime('%Y%m%d'))
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_train.loc[(curr_date,slice(None),:)]
        self.tickers = self.X_test.index.tolist()
        self.result = pd.DataFrame(columns =['PRED','SnpDate'],index = self.tickers)
        self.result['SnpDate'] = self.snp_date
        
    def predict_date(self,curr_date):

        for ticker in self.tickers:

            x_train = self.X_train.loc[(slice(None,ticker),:)].values()
            y_train = self.Y_train.loc[(slice(None,ticker),:)].values()
            x_test = self.X_test.loc[ticker].values()
            model_ticker = GradientBoostingRegressor(loss ='absolute_error',
                                                    n_estimators = 50,
                                                    max_depth = 8,
                                                    max_features = 20
                                                    )
            model_ticker.fit(x_train,y_train)

            y_pred = model_ticker.predict(x_test.reshape(1,-1))[0]
            self.result.loc[ticker,'PRED'] = y_pred

    def export_step(self):
        # create a pyarrow table from the pandas dataframe
        table = pa.Table.from_pandas(self.result)

        # define the partitioning schema
        partition_schema = pa.schema([("SnpDate", pa.int64())])

        # define the output directory path
        output_dir = 'model/'+self.name_model+'/'
        # write the table as a partitioned parquet file
        pq.write_to_dataset(table,
                            root_path=output_dir,
                            partition_cols=["SnpDate"],
                            partition_filename_cb=lambda partition_keys: f"{partition_keys[0]}.parquet",
                            partition_schema=partition_schema
                            )
        return 
    
    


