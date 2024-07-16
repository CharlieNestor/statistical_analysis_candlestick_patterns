import numpy as np
import pandas as pd
import patterns as pt
import statsmodels.api as sm
import statsmodels.stats.api as sms
from scipy import stats


from typing import List, Dict, Tuple, Union



def add_pct_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the percentage and log returns of the stock price
    """
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(1 + df['Returns'])
    return df


# TECHNICAL INDICATORS functions

def calculate_ATR(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate the True Range (TR) and Average True Range (ATR) for a given DataFrame.
    : param df: DataFrame with OHLC data.
    : param period: Period for calculating the ATR (default is 14).
    : return: New DataFrame with TR and ATR columns.
    """
    # ensure the DataFrame contains the necessary columns
    if not {'High', 'Low', 'Close'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns.")
    
    new_df = pd.DataFrame(index=df.index)

    # calculate the True Range (TR)
    df['Prev_Close'] = df['Close'].shift(1)
    new_df['TR'] = df[['High', 'Low', 'Prev_Close']].apply(
        lambda row: max(row['High'] - row['Low'], 
                        abs(row['High'] - row['Prev_Close']), 
                        abs(row['Low'] - row['Prev_Close'])), axis=1)

    # calculate the initial ATR as the rolling mean of the first 'period' TR values
    new_df['ATR'] = new_df['TR'].rolling(window=period).mean()

    # calculate subsequent ATR values using the formula:
    # ATR(i) = (ATR(i-1) * (period - 1) + TR(i)) / period
    for i in range(period+1, len(new_df)):
        #new_df.iloc[i].at['ATR'] = (new_df.iloc[i-1].at['ATR'] * (period - 1) + new_df.iloc[i].at['TR']) / period
        new_df.loc[new_df.index[i], 'ATR'] = (new_df.iloc[i-1]['ATR'] * (period - 1) + new_df.iloc[i]['TR']) / period

    # Drop the intermediate 'Prev_Close' column
    df.drop(columns=['Prev_Close'], inplace=True)

    return new_df


# STATISTICAL TESTS


def normality_tests(data: Union[pd.Series, np.ndarray]) -> dict:
    """
    Perform normality tests on the given data.
    :param data: The data to test for normality.
    :return: A dictionary containing the results of the Kolmogorov-Smirnov, Shapiro-Wilk, and Jarque-Bera tests.
    """
    # Convert to numpy array if it's a pandas Series
    if isinstance(data, pd.Series):
        data = data.values
    
    # Ensure the data is 1-dimensional
    if data.ndim > 1:
        raise ValueError("Input data must be 1-dimensional")

    # Remove any NaN values
    data = data[~np.isnan(data)]

    # Perform Kolmogorov-Smirnov test
    ks_statistic, ks_pvalue = stats.kstest(data, 'norm')

    # Perform Shapiro-Wilk test
    sw_statistic, sw_pvalue = stats.shapiro(data)

    # Perform Jarque-Bera test
    jb_statistic, jb_pvalue = stats.jarque_bera(data)

    # Create a dictionary with the results
    results = {
        'Kolmogorov-Smirnov': {'statistic': ks_statistic, 'p-value': ks_pvalue},
        'Shapiro-Wilk': {'statistic': sw_statistic, 'p-value': sw_pvalue},
        'Jarque-Bera': {'statistic': jb_statistic, 'p-value': jb_pvalue}
    }

    return results


def check_metric_normality(metrics: Dict[str, Dict[int, np.ndarray]], metric_name: str, verbose: bool = True) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Check the normality of a specific metric's distribution for each day.
    : param metrics: The metrics dictionary output from calculate_metrics function. Keys are metric names. Values are dictionaries with days as keys and numpy arrays of metric values as values.
    : param metric_name: The name of the metric to check for normality.
    : param verbose: Whether to print the results (default is True).
    : return: A dictionary with days as keys and normality test results as values.
    """
    if metric_name not in metrics:
        raise ValueError(f"Metric '{metric_name}' not found in the metrics dictionary.")

    normality_results = {}

    for day, values in metrics[metric_name].items():
        day_results = normality_tests(values)
        normality_results[day] = day_results

        if verbose:
            print(f"\nNormality tests for {metric_name} on Day {day}:")
            print("=" * 50)
            for test_name, test_results in day_results.items():
                print(f"{test_name} Test:")
                print(f"  Statistic: {test_results['statistic']:.4f}")
                print(f"  p-value: {test_results['p-value']:.4f}")
            print()

    return normality_results




# ANALYSIS functions relative to each periods (1 to 10 days) and metrics

def calculate_cumReturns_periods(df: pd.DataFrame, pattern_mask: pd.Series, max_ahead = 10) -> dict[int, list[float]]:
    """
    Calculate the cumulative returns following a pattern for different future periods / candles
    :param df: the stock dataset
    :param pattern_mask: a boolean mask with True where patterns occur
    :param max_ahead: the maximum number of periods to calculate the returns
    :return: a dictionary with the cumulative returns for each period. Keys are the periods ( = 1,2,3,4,...), values are lists of returns
    """
    periods = range(1, max_ahead + 1)
    returns = {period: [] for period in periods}
    for i in range(1, len(df) - max(periods)):      # start from 1 since the first row might have NaN
        if pattern_mask.iloc[i]:                # the pattern occurs at this date
            for period in periods:              # calculate the returns for each period
                if i + period < len(df):
                    # simple cumulative return formula
                    returns[period].append((df['Close'].iloc[i + period] - df['Close'].iloc[i]) / df['Close'].iloc[i])
    
    return returns


def calculate_log_cumReturns_periods(df: pd.DataFrame, pattern_mask: pd.Series, max_ahead = 10) -> dict[int, list[float]]:
    """
    Calculate the cumulative log returns following a pattern for different future periods / candles
    :param df: the stock dataset
    :param pattern_mask: a boolean mask with True where patterns occur
    :param max_ahead: the maximum number of periods to calculate the returns
    :return: a dictionary with the cumulative log returns for each period. Keys are the periods ( = 1,2,3,4,...), values are lists of log returns
    """
    periods = range(1, max_ahead + 1)
    returns = {period: [] for period in periods}
    for i in range(1, len(df) - max(periods)):      # start from 1 since the first row might have NaN
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
    :return:    a dictionary with the win rate for each period. 
                Keys are the periods ( = 1,2,3,...), values are the win rate rounded to 2 decimal places
    """
    win_rate = {period: round(sum(1 for r in ret if r > 0) * 100 / len(ret), 2) for period, ret in returns.items() if ret}
    return win_rate

def calculate_average_return(returns: dict[int, list[float]]) -> dict[int, float]:
    """
    Calculate the average return for each future period
    :param returns: a dictionary with the cumulative returns for each period.
    :return:    a dictionary with the average return for each period. 
                Keys are the periods ( = 1,2,3,...), values are the average return rounded to 3 decimal places
    """
    avg_return = {period: round(sum(ret) / len(ret), 3) for period, ret in returns.items() if ret}
    return avg_return

def calculate_median_return(returns: dict[int, list[float]]) -> dict[int, float]:
    """
    Calculate the median return for each future period
    :param returns: a dictionary with the cumulative returns for each period.
    :return:    a dictionary with the median return for each period. 
                Keys are the periods ( = 1,2,3,...), values are the median return rounded to 3 decimal places
    """
    median_return = {period: round(np.nanmedian(ret),3) for period, ret in returns.items() if ret}
    return median_return


def calculate_returns_pattern(series: pd.Series, pattern_mask: pd.Series, max_length: int = 100) -> list[np.ndarray]:
    """
    Calculate the cumulative returns following a pattern for different future periods / candles
    :param series: the stock price series
    :param pattern_mask: a boolean mask with True where patterns occur
    :param max_length: the maximum length of observation after the pattern
    :return: a list of percentage returns for each pattern occurrence for the next 'max_length' days
    """
    return_series = []
    for i in range(1, len(series) - max_length):        # start from 1 since the first row might have NaN
        if pattern_mask.iloc[i]:                    # the pattern occurs at this date
            return_series.append(series.iloc[i:i+max_length])
    
    return return_series



def generate_multiple_mask(df: pd.DataFrame, input_mask, dim_sample: int, n_iterations: int = 1000, lag: int = 10):
    """
    Generate multiple random masks for a DataFrame.
    :param df: DataFrame to generate masks for
    :param input_mask: Mask to use as input for generating random masks
    :param dim_sample: Number of samples to generate
    :param n_iterations: Number of masks to generate
    :param lag: Minimum separation between samples
    :return: List of random masks
    """
    return [pt.random_mask(df = df, input_mask=input_mask, dim_sample = dim_sample) for _ in range(n_iterations)]


def generate_random_returns(df: pd.DataFrame, input_mask: pd.Series, dim_sample: int, n_iterations: int = 1000, verbose: bool = True) -> list[dict[int, np.ndarray[float]]]:
    """
    Generate random returns from random masks. Returns will be rounded to 3 decimal places.
    :param df: DataFrame to generate returns for
    :param input_mask: Mask to use as input for generating random masks
    :param n_iterations: Number of random samples to generate
    :param dim_sample: Dimension of each random sample
    :return: List of random returns. The returns are dictionaries with keys as periods and values as numpy arrays of returns
    """
    random_masks = generate_multiple_mask(df, input_mask, dim_sample=dim_sample, n_iterations=n_iterations)
    original_returns = []
    counter = 0
    if verbose:
        print('Starting generating samples...')
    for mask in random_masks:
        returns = calculate_cumReturns_periods(df, mask, max_ahead=15)
        returns = {k: np.array([round(100*r,3) for r in v]) for k, v in returns.items()}      # round to 3 decimal places
        original_returns.append(returns)
        counter += 1
        if verbose:
            print(f'Generated {counter} samples.')

    return original_returns


def calculate_metrics(samples: List[Dict[int, np.ndarray]]) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Calculate win rate, average return, and median return for each day across all samples.
    
    :param samples: List of dictionaries, where each dictionary represents a sample
                    with keys as days (1-10) and values as numpy arrays of returns.
    :return: Dictionary with metrics as keys and nested dictionaries as values.
             The nested dictionaries have days as keys and numpy arrays of metric values as values.
    """
    n_days = len(samples[0])
    n_samples = len(samples)
    
    # Initialize the results dictionary
    results = {
        'win_rate': {},
        'average_return': {},
        'median_return': {},
        'std_return': {},
        'kurtosis_return': {}
    }
    
    for day in range(1, n_days + 1):
        win_rates = []
        avg_returns = []
        median_returns = []
        std_returns = []
        kurtosis_returns = []
        
        for sample in samples:
            day_returns = sample[day]
            
            # Calculate win rate
            win_rate = (day_returns > 0).sum() / len(day_returns) * 100
            win_rates.append(win_rate)
            
            # Calculate average return
            avg_return = np.mean(day_returns)
            avg_returns.append(avg_return)
            
            # Calculate median return
            median_return = np.median(day_returns)
            median_returns.append(median_return)

            # Calculate the standard deviation of returns
            std_return = np.std(day_returns)
            std_returns.append(std_return)

            # Calculate the kurtosis of returns
            kurtosis_return = stats.kurtosis(day_returns)
            kurtosis_returns.append(kurtosis_return)

        
        # Store the results for this day
        results['win_rate'][day] = np.array(win_rates)
        results['average_return'][day] = np.array(avg_returns)
        results['median_return'][day] = np.array(median_returns)
        results['std_return'][day] = np.array(std_returns)
        results['kurtosis_return'][day] = np.array(kurtosis_returns)
    
    return results


def calculate_confidence_intervals(metrics: Dict[str, Dict[int, np.ndarray]], low_perc: float = 2.5, high_perc: float = 97.5) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """
    Calculate the 95% confidence intervals for each metric based on the given distributions.
    
    :param metrics: Dictionary with metrics as keys and nested dictionaries as values.
                    The nested dictionaries have days as keys and numpy arrays of metric values as values.
    :return: Dictionary with metrics as keys and nested dictionaries as values.
             The nested dictionaries have days as keys and tuples of confidence intervals (lower, upper) as values.
    """
    confidence_intervals = {}
    
    for metric, day_values in metrics.items():
        confidence_intervals[metric] = {}
        
        for day, values in day_values.items():
            lower = np.percentile(values, low_perc)
            upper = np.percentile(values, high_perc)
            confidence_intervals[metric][day] = (round(lower, 3), round(upper, 3))
    
    return confidence_intervals

