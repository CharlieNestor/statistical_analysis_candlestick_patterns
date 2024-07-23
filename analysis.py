import math
import numpy as np
import pandas as pd
import patterns as pt
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
    :return: a dictionary with the win rate for each period. 
            Keys are the periods ( = 1,2,3,...), values are the win rate rounded to 2 decimal places
    """
    win_rate = {period: round(sum(1 for r in ret if r > 0) * 100 / len(ret), 2) for period, ret in returns.items() if ret}
    return win_rate

def calculate_average_return(returns: dict[int, list[float]]) -> dict[int, float]:
    """
    Calculate the average return for each future period
    :param returns: a dictionary with the cumulative returns for each period.
    :return: a dictionary with the average return for each period. 
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


def log_to_simple(log_return: Union[float, pd.Series, np.ndarray[float]]) -> float:
    """
    Convert log returns to simple returns and multiply by 100 to get percentage returns in readable format.
    """
    return (np.exp(log_return) - 1) * 100


def calculate_win_rate_from_log(log_returns: dict[int, list[float]]) -> dict[int, float]:
    """
    Calculate the win rate for each future period as the percentage of positive log returns in each period.
    :param log_returns: a dictionary with the cumulative log returns for each period.
    :return: a dictionary with the win rate for each period.
             Keys are the periods (1,2,3,...), values are the win rate rounded to 2 decimal places.
    """
    win_rate = {period: round(sum(1 for r in ret if r > 0) * 100 / len(ret), 2) 
                for period, ret in log_returns.items() if ret}
    return win_rate

def calculate_average_return_from_log(log_returns: dict[int, list[float]]) -> dict[int, float]:
    """
    Calculate the average return for each future period from log returns, converted to linear scale.
    :param log_returns: a dictionary with the cumulative log returns for each period.
    :return: a dictionary with the average return for each period in linear scale.
             Keys are the periods (1,2,3,...), values are the average return rounded to 3 decimal places.
    """
    avg_return = {period: round(log_to_simple(np.mean(ret)), 3) 
                  for period, ret in log_returns.items() if ret}
    return avg_return

def calculate_median_return_from_log(log_returns: dict[int, list[float]]) -> dict[int, float]:
    """
    Calculate the median return for each future period from log returns, converted to linear scale.
    :param log_returns: a dictionary with the cumulative log returns for each period.
    :return: a dictionary with the median return for each period in linear scale.
             Keys are the periods (1,2,3,...), values are the median return rounded to 3 decimal places.
    """
    median_return = {period: round(log_to_simple(np.median(ret)), 3) 
                     for period, ret in log_returns.items() if ret}
    return median_return

def calculate_std_return_from_log(log_returns: dict[int, list[float]]) -> dict[int, float]:
    """
    Calculate the standard deviation of returns for each future period from log returns, converted to linear scale.
    :param log_returns: a dictionary with the cumulative log returns for each period.
    :return: a dictionary with the standard deviation of returns for each period in linear scale.
             Keys are the periods (1,2,3,...), values are the standard deviation rounded to 3 decimal places.
    """
    std_return = {period: round(log_to_simple(np.std(ret)), 3) 
                  for period, ret in log_returns.items() if ret}
    return std_return


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


# GENERATE RANDOM SAMPLES functions


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


def generate_random_returns(df: pd.DataFrame, input_mask: pd.Series, dim_sample: int, n_iterations: int = 1000, 
                            is_log: bool = True, verbose: bool = True) -> list[dict[int, np.ndarray[float]]]:
    """
    Generate random returns from random masks. Returns will be rounded to 3 decimal places.
    :param df: DataFrame to generate returns for
    :param input_mask: Mask to use as input for generating random masks
    :param n_iterations: Number of random samples to generate
    :param dim_sample: Dimension of each random sample
    :param is_log: Whether to calculate log returns
    :param verbose: Whether to print progress messages
    :return: List of random returns. The returns are dictionaries with keys as periods and values as numpy arrays of returns
    """
    # Ensure the dimension of the sample is a multiple of 50 with a minimum of 100
    floor = 100
    if dim_sample <= floor:
            dim_sample = floor
    else:
        dim_sample = math.ceil(dim_sample / 50) * 50
    # Generate random masks
    random_masks = generate_multiple_mask(df, input_mask, dim_sample=dim_sample, n_iterations=n_iterations)
    original_returns = []
    counter = 0
    if verbose:
        print('Starting generating samples...')
    for mask in random_masks:
        if is_log:
            returns = calculate_log_cumReturns_periods(df, mask, max_ahead=15)
            returns = {k: np.array([r for r in v]) for k, v in returns.items()}      # convert to numpy array
        else:
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

        
        # Store the results for this day
        results['win_rate'][day] = np.array(win_rates)
        results['average_return'][day] = np.array(avg_returns)
        results['median_return'][day] = np.array(median_returns)
        results['std_return'][day] = np.array(std_returns)
    
    return results


# STATISTICAL TESTS functions


def calculate_nonParametric_confidence_intervals(metrics: Dict[str, Dict[int, np.ndarray]], 
                                                low_perc: float = 2.5, high_perc: float = 97.5, 
                                                is_log_ret: bool = True) -> Dict[str, Dict[int, Tuple[float, float]]]:
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
            mean = np.mean(values)

            # if the returns are in log scale, convert the confidence intervals to linear scale
            if is_log_ret and metric != 'win_rate':     # not valid for win_rate
                # Convert from log scale to linear scale
                lower = (np.exp(lower) - 1) * 100
                upper = (np.exp(upper) - 1) * 100
                mean = (np.exp(mean) - 1) * 100         # Geometric mean in linear scale

            confidence_intervals[metric][day] = (round(lower, 3), round(mean, 3), round(upper, 3))
    
    return confidence_intervals


def calculate_parametric_confidence_intervals(metrics: Dict[str, Dict[int, np.ndarray]], 
                                              confidence_level: float = 0.95,
                                              is_log_ret: bool = True) -> Dict[str, Dict[int, Tuple[float, float, float]]]:
    """
    Calculate the parametric confidence intervals for each metric based on the given distributions.
    
    :param metrics: Dictionary with metrics as keys and nested dictionaries as values.
                    The nested dictionaries have days as keys and numpy arrays of metric values as values.
    :param confidence_level: Confidence level for the interval (default is 0.95 for 95% CI)
    :param is_log_ret: Boolean indicating whether the returns are in log scale (default is True)
    :return: Dictionary with metrics as keys and nested dictionaries as values.
             The nested dictionaries have days as keys and tuples of confidence intervals (lower, mean, upper) as values.
    """
    parametric_confidence_intervals = {}
    z_value = stats.norm.ppf((1 + confidence_level) / 2)  # z-value for the given confidence level

    for metric, day_values in metrics.items():
        parametric_confidence_intervals[metric] = {}
        for day, values in day_values.items():
            if is_log_ret and metric != 'win_rate':
                # For log returns
                mean = np.mean(values)
                std_error = np.std(values)
                
                # Calculate CI in log scale
                lower_log = mean - z_value * std_error
                upper_log = mean + z_value * std_error
                
                # Transform to linear scale
                lower = (np.exp(lower_log) - 1) * 100
                upper = (np.exp(upper_log) - 1) * 100
                mean = (np.exp(mean) - 1) * 100     # Geometric mean in linear scale
            else:
                # For win rate or if not using log returns
                mean = np.mean(values)
                std_error = np.std(values)
                lower = mean - z_value * std_error
                upper = mean + z_value * std_error

            parametric_confidence_intervals[metric][day] = (round(lower, 3), round(mean, 3), round(upper, 3))

    return parametric_confidence_intervals


def empirical_pvalue(base_distribution: np.ndarray, pattern_value: float, greater: bool = True) -> float:
    """
    Calculate the empirical p-value for a given pattern value and base distribution.
    
    :param base_distribution: Numpy array of base metric values.
    :param pattern_value: Single value from the pattern metric.
    :param greater: If True, calculate upper-tail p-value; if False, calculate lower-tail p-value.
    :return: Empirical p-value.
    """
    if greater:
        return (base_distribution >= pattern_value).mean()
    else:
        return (base_distribution <= pattern_value).mean()


def calculate_empirical_pvalues(pattern_metrics: Dict[str, Dict[int, float]], 
                                base_distributions: Dict[str, Dict[int, np.ndarray]], 
                                max_days: int = 15) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    Calculate empirical p-values comparing pattern metrics to base distribution metrics.

    This function computes both upper-tail (pattern > base) and lower-tail (pattern < base) 
    empirical p-values for each metric and day.

    :param pattern_metrics: Dictionary of pattern metrics. 
                            Keys are metric names, values are dictionaries with days as keys and metric values as values.
    :param base_distributions: Dictionary of base metrics. 
                         Keys are metric names, values are dictionaries with days as keys and numpy arrays of metric values as values.
    :param max_days: Maximum number of days to calculate p-values for (default is 15).
    :return: Dictionary with metrics as keys, containing nested dictionaries for 'high' and 'low' p-values, 
             each containing dictionaries with days as keys and p-values as values.
    """
    empirical_pvalues = {}

    for metric in pattern_metrics.keys():
        empirical_pvalues[metric] = {'high': {}, 'low': {}}
        
        for day in range(1, max_days + 1):
            base_values = base_distributions[metric][day]
            
            if metric != 'win_rate':  # win rate is already in linear scale
                pattern_value = pattern_metrics[metric][day] / 100  # Convert percentage to decimal
            else:
                pattern_value = pattern_metrics[metric][day]
            
            p_value_greater = empirical_pvalue(base_values, pattern_value, greater=True)
            p_value_less = empirical_pvalue(base_values, pattern_value, greater=False)
            
            empirical_pvalues[metric]['high'][day] = p_value_greater
            empirical_pvalues[metric]['low'][day] = p_value_less

    return empirical_pvalues


def transform_confidence_intervals(ci_dict: Dict[str, Dict[int, Tuple[float, float, float]]]) -> Tuple[Dict[str, Dict[str, Dict[int, float]]], Dict[str, Dict[int, float]]]:
    """
    Transform the structure of confidence interval dictionaries.

    This function takes a dictionary of confidence intervals and transforms it into two separate dictionaries:
    one for the confidence intervals (with 'high' and 'low' bounds) and another for the mean estimates.

    :param ci_dict: A dictionary with metrics as keys, containing nested dictionaries with days as keys 
                    and tuples (lower bound, mean, upper bound) as values.
    :return: A tuple containing two dictionaries:
             1. A transformed dictionary with metrics as keys, containing nested dictionaries for 'high' and 'low' bounds,
                each containing dictionaries with days as keys and bound values as values.
             2. A dictionary with metrics as keys, containing nested dictionaries with days as keys and mean values as values.
    """
    transformed = {}
    basecase_estimates = {}

    for metric, days_data in ci_dict.items():
        transformed[metric] = {'high': {}, 'low': {}}
        basecase_estimates[metric] = {}

        for day, (lower, mean, upper) in days_data.items():
            transformed[metric]['low'][day] = lower
            transformed[metric]['high'][day] = upper
            basecase_estimates[metric][day] = mean

    return transformed, basecase_estimates


def create_significance_table(pattern_metrics: Dict[str, Dict[int, float]], 
                              basecase_estimates: Dict[str, Dict[int, float]],
                              empirical_pvalues: Dict[str, Dict[str, Dict[int, float]]],
                              parametric_conf_int: Dict[str, Dict[str, Dict[int, float]]],
                              non_parametric_conf_int: Dict[str, Dict[str, Dict[int, float]]],) -> Dict[str, np.ndarray]:
    """
    Create a table of significance levels for different metrics comparing pattern performance to base case.
    
    This function compares pattern metrics to base case estimates and various confidence intervals,
    assigning significance scores based on the comparisons. Positive scores indicate the pattern
    outperforming the base case, while negative scores indicate underperformance.
    
    :param pattern_metrics: Dictionary of pattern metric values
    :param basecase_estimates: Dictionary of base case estimate values
    :param empirical_pvalues: Dictionary of empirical p-values
    :param parametric_conf_int: Dictionary of parametric confidence intervals
    :param non_parametric_conf_int: Dictionary of non-parametric confidence intervals
    :return: Dictionary with metrics as keys and numpy arrays of significance scores as values
    """

    metrics_list = ['win_rate', 'average_return', 'median_return']
    days = len(pattern_metrics['win_rate'])
    table = {metric: np.zeros(days, dtype=int) for metric in metrics_list}
    
    for metric in metrics_list:
        for day in range(1, days + 1):
            pattern_value = pattern_metrics[metric][day]
            basecase_value = basecase_estimates[metric][day]
            difference = pattern_value - basecase_value
            
            if difference > 0:
                direction = 'high'
                increment = 1
            else:
                direction = 'low'
                increment = -1
            
            count = 0
            # Check only the relevant direction (high or low) for all tests
            if empirical_pvalues[metric][direction][day] < 0.05:
                count += increment
            
            if direction == 'high':
                if pattern_value > parametric_conf_int[metric][direction][day]:
                    count += increment
                if pattern_value > non_parametric_conf_int[metric][direction][day]:
                    count += increment
            else:  # direction == 'low'
                if pattern_value < parametric_conf_int[metric][direction][day]:
                    count += increment
                if pattern_value < non_parametric_conf_int[metric][direction][day]:
                    count += increment
            
            table[metric][day - 1] = count  # -1 because array is 0-indexed
    
    return table