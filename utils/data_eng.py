import pandas_ta as ta
from utils.basic_features import (
    signalMACD,
    compute_returns,
    compute_volatility_daily,
    compute_returns_vol_adjusted,
    compute_beta,
)
import numpy as np
import pandas as pd
import os


MKT_RETURNS = pd.read_csv("data/US/mkt_returns.csv", index_col=0)
MKT_RETURNS.index = pd.to_datetime(MKT_RETURNS.index)
SQUEEZE_PARAM = 5
OUTLIERS_TIME_LAG = 252


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
