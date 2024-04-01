import numpy as np
import pandas as pd

from typing import Dict, List, Tuple

from empyrical import (
    max_drawdown,
    downside_risk,
    sortino_ratio,
    annual_return,
    annual_volatility,
    sharpe_ratio,
    calmar_ratio,
)

VOL_LOOKBACK = 60  # for ex-ante volatility
VOL_TARGET = 0.15  # 15% volatility target


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
