import argparse
import datetime as dt
from typing import List
from tqdm import tqdm
import pandas as pd
import numpy as np
from tqdm import tqdm
import pandas_ta as ta
from utils.data_eng import features_for_model
from typing import Tuple,List


VOL_LOOKBACK = 60  # for ex-ante volatility
VOL_TARGET = 0.15  # 15% volatility target
MKT_RETURNS = pd.read_csv("data/US/mkt_returns.csv", index_col=0)
MKT_RETURNS.index = pd.to_datetime(MKT_RETURNS.index)
SQUEEZE_PARAM = 5
OUTLIERS_TIME_LAG = 252


def compute_returns(price: pd.Series, day_offset: int = 1) -> pd.Series:
    """
    Calculates the returns of a given price series over a specified number of days.

    Args:
        price (pd.Series): Time series of prices.
        day_offset (int, optional): Number of days to calculate returns. Defaults to 1.

    Returns:
        pd.Series: Series of calculated returns.
    """
    returns = price / price.shift(day_offset) - 1.0
    return returns


def compute_volatility_daily(daily_returns, vol_lookback=VOL_LOOKBACK):
    """
    Computes the daily exponential moving average (EMA) volatility.

    Args:
        daily_returns: Daily return series.
        vol_lookback (int, optional): Lookback period for volatility calculation. Defaults to VOL_LOOKBACK.

    Returns:
        pd.Series: EMA of daily volatility.
    """
    return (
        daily_returns.ewm(span=vol_lookback, min_periods=vol_lookback)
        .std()
        .ffill()
    )


def compute_returns_vol_adjusted(returns, vol=pd.Series(None), annualization=252):
    """
    Adjusts returns for volatility targeting an annual volatility level.

    Args:
        returns: Series of returns.
        vol (pd.Series, optional): Volatility series. If not provided, it is computed. Defaults to None.
        annualization (int, optional): Annualization factor. Defaults to 252.

    Returns:
        pd.Series: Volatility-adjusted returns.
    """
    if not len(vol):
        vol = compute_volatility_daily(returns)
    vol = vol * np.sqrt(annualization)
    return returns * VOL_TARGET / vol.shift(1)


def compute_beta(returns: pd.Series, mkt_returns: pd.Series, window=252):
    """
    Calculates the beta of a stock relative to market returns.

    Args:
        returns (pd.Series): Stock return series.
        mkt_returns (pd.Series): Market return series.
        window (int, optional): Rolling window for calculation. Defaults to 252.

    Returns:
        pd.Series: Beta values.
    """
    beta = returns.join(mkt_returns)
    beta.columns = ["stock", "mkt"]
    df_cov = beta.rolling(window).cov().unstack()["stock"]["mkt"]
    df_var = beta["mkt"].to_frame().rolling(window).var()
    return (df_cov / (df_var.T)).T["mkt"]


class signalMACD:
    """
    Class for computing Moving Average Convergence Divergence (MACD) signal on a price series.

    Methods:
        macd: Calculates the MACD signal for specified short and long timescales.
        calc_combined_signal: Aggregates MACD signals across different time scale combinations.
    """

    def __init__(self, long_short: List[Tuple[float, float]] = None):
        """
        class for computing MACD long short signal on a time series of price

        """
        if long_short is None:
            self.long_short = [(8, 24), (16, 48), (32, 96)]
        else:
            self.long_short = long_short

    @staticmethod
    def macd(price: pd.Series, short_timescale: int, long_timescale: int) -> float:
        """
        Args:
            price ([type]): series of prices
            short_timescale ([type]): short timescale
            long_timescale ([type]): long timescale

        Returns:
            float: MACD signal
        """

        def _calc_halflife(timescale):
            return np.log(0.5) / np.log(1 - 1 / timescale)

        macd = (
            price.ewm(halflife=_calc_halflife(short_timescale)).mean()
            - price.ewm(halflife=_calc_halflife(long_timescale)).mean()
        )
        q = macd / price.rolling(63).std().ffill()
        return q / q.rolling(252).std().ffill()

    def calc_combined_signal(self, price: pd.Series) -> float:
        """Combined MACD signal

        Argument:
            price  series of prices for a ticker
        Returns:
            macd signals for different combinaison of long and short time
        """
        return np.sum(
            [self.macd(price, short, long) for short, long in self.long_short]
        ) / len(self.long_short)



def center(x):
    mean = x.mean(1)
    x = x.sub(mean, 0)
    return x


def features_for_model(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Processes and generates a comprehensive set of features for a given stock data.

    This function applies a series of transformations and calculations to raw stock data to prepare
    it for use in a financial model. It includes standard and custom financial indicators, normalization,
    and other statistical transformations.

    Args:
        stock_data (pd.DataFrame): The raw stock data as a DataFrame.

    Returns:
        pd.DataFrame: The DataFrame enriched with additional financial features.

    Notes:
        - Filters out rows where the 'close' price is unavailable or practically zero.
        - Applies outlier handling, normalization, and calculation of various financial indicators.
        - Calculates beta values, alpha returns, MACD signals, and other statistical measures.
        - Adds time-related features like month and year.
    """
    stock_data = stock_data[
        ~stock_data["close"].isna()
        | ~stock_data["close"].isnull()
        | (stock_data["close"] > 1e-8)  # price is basically null
    ].copy()

    stock_data["srs"] = stock_data["close"]
    ewm = stock_data["srs"].ewm(halflife=OUTLIERS_TIME_LAG)
    means = ewm.mean()
    stds = ewm.std()
    stock_data["srs"] = np.minimum(stock_data["srs"], means + SQUEEZE_PARAM * stds)
    stock_data["srs"] = np.maximum(stock_data["srs"], means - SQUEEZE_PARAM * stds)

    stock_data["daily_returns"] = compute_returns(stock_data["srs"])
    stock_data["daily_vol"] = compute_volatility_daily(stock_data["daily_returns"])
    stock_data["weekly_returns"] = compute_returns(stock_data["srs"], day_offset=5) / 5
    stock_data["monthly_returns"] = compute_returns(stock_data["srs"], day_offset=21) / 21
    stock_data["quarterly_returns"] = compute_returns(stock_data["srs"], day_offset=63) / 63
    stock_data["biannual_returns"] = compute_returns(stock_data["srs"], day_offset=126) / 126
    stock_data["target_returns"] = compute_returns_vol_adjusted(
        returns=stock_data["daily_returns"],
        vol=stock_data["daily_vol"],
        annualization=252,
    ).shift(-1)

    stock_data["daily_vol_21"] = compute_volatility_daily(
        stock_data["daily_returns"], vol_lookback=21
    )
    stock_data["daily_vol_5"] = compute_volatility_daily(
        stock_data["daily_returns"], vol_lookback=5
    )

    def return_normalised(day_offset):
        """
        Normalizes returns over a specified offset period.

        Args:
        day_offset (int): The number of days over which to normalize the returns.

        Returns:
        pd.Series: Normalized return series.
        """
        return (
            compute_returns(stock_data["srs"], day_offset)
            / stock_data["daily_vol"]
            / np.sqrt(day_offset)
        )

    stock_data["norm_daily_return"] = return_normalised(1)
    stock_data["norm_weekly_return"] = return_normalised(5)
    stock_data["norm_monthly_return"] = return_normalised(21)
    stock_data["norm_quarterly_return"] = return_normalised(63)
    stock_data["norm_biannual_return"] = return_normalised(126)
    stock_data["norm_annual_return"] = return_normalised(252)
    stock_data["norm_twoannual_return"] = return_normalised(252*2)
    stock_data["norm_threeannual_return"] = return_normalised(252*3)

    stock_data["beta_252"] = compute_beta(
        stock_data[["daily_returns"]], MKT_RETURNS[["mkt_returns"]], window=252
    )

    stock_data["beta_126"] = compute_beta(
        stock_data[["daily_returns"]], MKT_RETURNS[["mkt_returns"]], window=126
    )
    
    stock_data["rsi_252"] = ta.rsi(stock_data["srs"], 252)
    stock_data["rsi_126"] = ta.rsi(stock_data["srs"], 126)
    stock_data["rsi_63"] = ta.rsi(stock_data["srs"], 63)
    stock_data["rsi_21"] = ta.rsi(stock_data["srs"], 21)
    stock_data["cti_252"] = ta.cti(stock_data["daily_returns"], 252)
    stock_data["cti_126"] = ta.cti(stock_data["daily_returns"], 126)
    stock_data["cti_63"] = ta.cti(stock_data["daily_returns"], 63)
    stock_data["cti_21"] = ta.cti(stock_data["daily_returns"], 21)

    def compute_alpha_return():
        """
        Computes the alpha returns of the stock relative to the market.

        This function calculates the alpha (excess returns over the market) based on daily returns and beta values.

        Returns:
            pd.Series: Alpha return series.
        """
        alpha = stock_data[["daily_returns", "beta_126"]].join(MKT_RETURNS, how="left")
        alpha["alpha_daily_return"] = (
            alpha["daily_returns"] - alpha["beta_126"] * alpha["mkt_returns"]
        )
        return alpha["alpha_daily_return"]

    stock_data["alpha_daily_returns"] = compute_alpha_return()
    stock_data["alpha_weekly_returns"] = (
        1 + stock_data["alpha_daily_returns"].rolling(5).sum()
    ) ** (1 / 5) - 1
    stock_data["alpha_monthly_returns"] = (
        1 + stock_data["alpha_daily_returns"].rolling(21).sum()
    ) ** (1 / 21) - 1

    stock_data["specific_variance"] = (
        stock_data["alpha_daily_returns"].rolling(window=252).var()
    )
    # Calculate total variance (variance of stock's daily returns)
    stock_data["total_variance"] = stock_data["daily_returns"].rolling(window=252).var()
    # Calculate Alpha Variance Ratio
    stock_data["annual_alpha_variance_ratio"] = (
        stock_data["specific_variance"] / stock_data["total_variance"]
    )

    macd_short_long = [(5,15),(10,30),(21,63),(63,126)]
    for short, long in macd_short_long:
        stock_data[f"macd_{short}_{long}"] = signalMACD.macd(
            stock_data["srs"], short, long
        )
    stock_data["month_of_year"] = stock_data.index.month
    stock_data["year"] = stock_data.index.year
    stock_data["date"] = stock_data.index  # duplication but sometimes makes life easier

    return stock_data.dropna()

def aggregate_features_model(close:pd.DataFrame):
    tickers = close.columns.tolist()
    all = []
    for ticker in tqdm(tickers):
        test = close[[ticker]]
        test.columns = ['close']
        features = features_for_model_pca(test)
        features['Ticker'] = ticker
        all.append(features)

    return pd.concat(all,axis=0).set_index('Ticker',append= True)


def features_for_model_pca(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Processes and generates a comprehensive set of features for a given stock data.

    This function applies a series of transformations and calculations to raw stock data to prepare
    it for use in a financial model. It includes standard and custom financial indicators, normalization,
    and other statistical transformations.

    Args:
        stock_data (pd.DataFrame): The raw stock data as a DataFrame.

    Returns:
        pd.DataFrame: The DataFrame enriched with additional financial features.

    Notes:
        - Filters out rows where the 'close' price is unavailable or practically zero.
        - Applies outlier handling, normalization, and calculation of various financial indicators.
        - Calculates beta values, alpha returns, MACD signals, and other statistical measures.
        - Adds time-related features like month and year.
    """
    stock_data = stock_data[
        ~stock_data["close"].isna()
        | ~stock_data["close"].isnull()
        | (stock_data["close"] > 1e-8)  # price is basically null
    ].copy()

    stock_data["srs"] = stock_data["close"]
    ewm = stock_data["srs"].ewm(halflife=OUTLIERS_TIME_LAG)
    means = ewm.mean()
    stds = ewm.std()
    stock_data["srs"] = np.minimum(stock_data["srs"], means + SQUEEZE_PARAM * stds)
    stock_data["srs"] = np.maximum(stock_data["srs"], means - SQUEEZE_PARAM * stds)

    stock_data["daily_returns"] = compute_returns(stock_data["srs"])
    # stock_data["daily_vol"] = compute_volatility_daily(stock_data["daily_returns"])
    # stock_data["weekly_returns"] = compute_returns(stock_data["srs"], day_offset=5) / 5
    stock_data["mom_1m"] = compute_returns(stock_data["srs"], day_offset=21) /1* 21
    stock_data["mom_2m"] = compute_returns(stock_data["srs"], day_offset=2*21) /2* 21
    stock_data["mom_4m"] = compute_returns(stock_data["srs"], day_offset=4*21) / 4 * 21
    stock_data["mom_6m"] = compute_returns(stock_data["srs"], day_offset=6*21) / 6 * 21
    stock_data["mom_8m"] = compute_returns(stock_data["srs"], day_offset=8*21) / 8 * 21
    stock_data["mom_12m"] = compute_returns(stock_data["srs"], day_offset=12*21) / 12 * 21
    stock_data["mom_16m"] = compute_returns(stock_data["srs"], day_offset=16*21) / 16 * 21
    stock_data["mom_18m"] = compute_returns(stock_data["srs"], day_offset=18*21) / 18 * 21
    stock_data["mom_24m"] = compute_returns(stock_data["srs"], day_offset=24*21) / 24 * 21
    stock_data["mom_30m"] = compute_returns(stock_data["srs"], day_offset=30*21) / 30 * 21
    stock_data["mom_36m"] = compute_returns(stock_data["srs"], day_offset=36*21) / 36 * 21
    stock_data["mom_42m"] = compute_returns(stock_data["srs"], day_offset=42*21) / 42 * 21


    return stock_data.dropna()



if __name__ == '__main__':
    market_data = pd.read_parquet('data/US/test_mkt_data.pq')
    close = market_data['close'].unstack()
    market_features = aggregate_features_model(close)
    market_features.to_parquet('data/US/market_features_pca.pq')
    
