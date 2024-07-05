import statistics
import numpy as np
import pandas as pd


def add_pct_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the percentage and log returns of the stock price
    """
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(1 + df['Returns'])
    return df


def calculate_cumReturns_periods(df: pd.DataFrame, pattern_mask: pd.Series, periods=range(1, 11)) -> dict[int, list[float]]:
    """
    Calculate the cumulative returns following a pattern for different future periods / candles
    :param df: the stock dataset
    :param pattern_mask: a boolean mask with True where patterns occur
    :param periods: the range of periods to calculate the returns
    :return: a dictionary with the cumulative returns for each period. Keys are the periods ( = 1,2,3,4,...), values are lists of returns
    """
    returns = {period: [] for period in periods}
    for i in range(1, len(df) - max(periods)):      # start from 1 since the first row might have NaN
        if pattern_mask.iloc[i]:            # the pattern occurs at this date
            for period in periods:          # calculate the returns for each period
                if i + period < len(df):
                    # simple cumulative return formula
                    returns[period].append((df['Close'].iloc[i + period] - df['Close'].iloc[i]) / df['Close'].iloc[i])
    
    return returns


def calculate_log_cumReturns_periods(df: pd.DataFrame, pattern_mask: pd.Series, periods=range(1, 11)) -> dict[int, list[float]]:
    """
    Calculate the cumulative log returns following a pattern for different future periods / candles
    :param df: the stock dataset
    :param pattern_mask: a boolean mask with True where patterns occur
    :param periods: the range of periods to calculate the returns
    :return: a dictionary with the cumulative log returns for each period. Keys are the periods ( = 1,2,3,4,...), values are lists of log returns
    """
    returns = {period: [] for period in periods}
    for i in range(1, len(df) - max(periods)):    # start from 1 since the first row might have NaN
        if pattern_mask.iloc[i]:
            for period in periods:
                if i + period < len(df):
                    # log cumulative return formula
                    returns[period].append(np.log(df['Close'].iloc[i + period] / df['Close'].iloc[i]))

    return returns


def calculate_win_rate(returns: dict[int, list[float]]) -> dict[int, float]:
    """
    Calculate the win rate for each future period as the percentage of positive returns in each period
    :param returns: a dictionary with the cumulative returns for each period.
    :return: a dictionary with the win rate for each period. Keys are the periods ( = 1,2,3,...), values are the win rate rounded to 2 decimal places
    """
    win_rate = {period: round(sum(1 for r in ret if r > 0) *100 / len(ret), 2) for period, ret in returns.items() if ret}
    return win_rate

def calculate_average_return(returns: dict[int, list[float]]) -> dict[int, float]:
    """
    Calculate the average return for each future period
    :param returns: a dictionary with the cumulative returns for each period.
    :return: a dictionary with the average return for each period. Keys are the periods ( = 1,2,3,...), values are the average return rounded to 3 decimal places
    """
    avg_return = {period: round(sum(ret) / len(ret), 3) for period, ret in returns.items() if ret}
    return avg_return

def calculate_median_return(returns: dict[int, list[float]]) -> dict[int, float]:
    """
    Calculate the median return for each future period
    :param returns: a dictionary with the cumulative returns for each period.
    :return: a dictionary with the median return for each period. Keys are the periods ( = 1,2,3,...), values are the median return rounded to 3 decimal places
    """
    #median_return = {period: sorted(ret)[len(ret) // 2] for period, ret in returns.items() if ret}     # valid only for odd number of elements
    median_return = {period: round(statistics.median(ret),3) for period, ret in returns.items() if ret}
    return median_return