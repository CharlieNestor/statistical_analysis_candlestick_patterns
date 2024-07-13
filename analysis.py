import statistics
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import patterns as pt
import statsmodels.api as sm
import statsmodels.stats.api as sms
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import durbin_watson
from typing import List, Dict, Tuple



def add_pct_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the percentage and log returns of the stock price
    """
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(1 + df['Returns'])
    return df

# TECHNICAL INDICATORS functions

def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the log returns for a given DataFrame
    """
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(1 + df['Returns'])

    return df

def calculate_ATR(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate the True Range (TR) and Average True Range (ATR) for a given DataFrame.
    : param df: DataFrame with OHLC data.
    : param period: Period for calculating the ATR (default is 14).
    : return: DataFrame with 'TR' and 'ATR' columns added.
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

def adf_test(series: pd.Series) -> None:
    """
    Perform the Augmented Dickey-Fuller test on a time series.
    : param series: The time series to test
    : return: Three values containing the ADF statistic and p-value and the number of lags used in the test
    """
    result = adfuller(series.dropna(), maxlag=15)
    # Extract and display the test statistics
    adf_statistic = result[0]
    p_value = result[1]
    used_lag = result[2]
    
    print('Augmented Dickey-Fuller Test:')
    print(f"Statistic: {adf_statistic}")
    print(f"p-value: {p_value}")
    print(f"Used Lag: {used_lag}")

def ljung_box_test(series: pd.Series, lags: int = 10):
    result = acorr_ljungbox(series.dropna(), lags=lags)
    print('Ljung-Box Test:')
    print('Statistic:', result.lb_stat)
    print('p-value:', result.lb_pvalue)

def durbin_watson_test(series: pd.Series):
    dw_statistic = durbin_watson(series.dropna())
    print('Durbin-Watson Statistic:', dw_statistic)

def engle_arch_test(series: pd.Series, lags: int = 12):
    result = het_arch(series.dropna(), nlags=lags)
    print('Engle\'s ARCH Test:')
    print('Lagrange Multiplier statistic:', result[0])
    print('p-value:', result[1])

def jarque_bera_test(series: pd.Series):
    result = stats.jarque_bera(series.dropna())
    print('Jarque-Bera Test:')
    print('Statistic:', result.statistic)
    print('p-value:', result.pvalue)


def run_tests(series):
    results = {}
    
    # ADF Test
    adf_result = adfuller(series)
    try:
        results['adf_statistic'] = adf_result[0]
        results['adf_pvalue'] = adf_result[1]
    except Exception as e:
        results['adf_statistic'] = None
        results['adf_pvalue'] = None
        print(f"ADF test failed: {e}")
    
    # Ljung-Box Test
    lb_result = acorr_ljungbox(series, lags=10)
    try:
        lb_stats = lb_result.lb_stat
        lb_pvalues = lb_result.lb_pvalue
        for i in range(1,11):
            results[f'lb_statistic_lag_{i}'] = lb_stats[i]
            results[f'lb_pvalue_lag_{i}'] = lb_pvalues[i]
    except Exception as e:
        for i in range(1,11):
            results[f'lb_statistic_lag_{i}'] = None
            results[f'lb_pvalue_lag_{i}'] = None
        print(f"Ljung-Box test failed: {e}")
    
    # Durbin-Watson Test
    try:
        results['dw_statistic'] = durbin_watson(series)
    except Exception as e:
        results['dw_statistic'] = None
        print(f"Durbin-Watson test failed: {e}")
    
    # Engle's ARCH Test
    arch_result = het_arch(series)
    try:
        results['arch_statistic'] = arch_result[0]
        results['arch_pvalue'] = arch_result[1]
    except Exception as e:
        results['arch_statistic'] = None
        results['arch_pvalue'] = None
        print(f"Engle's ARCH test failed: {e}")
    
    # Jarque-Bera Test
    jb_result = stats.jarque_bera(series)
    try:
        results['jb_statistic'] = jb_result.statistic
        results['jb_pvalue'] = jb_result.pvalue
    except Exception as e:
        results['jb_statistic'] = None
        results['jb_pvalue'] = None
        print(f"Jarque-Bera test failed: {e}")
    
    return results



# ANALYSIS functions relative to each periods (1 to 10 days) and metrics

def calculate_cumReturns_periods(df: pd.DataFrame, pattern_mask: pd.Series, max_ahead = 10) -> dict[int, list[float]]:
    """
    Calculate the cumulative returns following a pattern for different future periods / candles
    :param df: the stock dataset
    :param pattern_mask: a boolean mask with True where patterns occur
    :param periods: the range of periods to calculate the returns
    :return: a dictionary with the cumulative returns for each period. Keys are the periods ( = 1,2,3,4,...), values are lists of returns
    """
    periods = range(1, max_ahead + 1)
    returns = {period: [] for period in periods}
    for i in range(1, len(df) - max(periods)):      # start from 1 since the first row might have NaN
        if pattern_mask.iloc[i]:            # the pattern occurs at this date
            for period in periods:          # calculate the returns for each period
                if i + period < len(df):
                    # simple cumulative return formula
                    returns[period].append((df['Close'].iloc[i + period] - df['Close'].iloc[i]) / df['Close'].iloc[i])
    
    return returns


def calculate_log_cumReturns_periods(df: pd.DataFrame, pattern_mask: pd.Series, max_ahead = 10) -> dict[int, list[float]]:
    """
    Calculate the cumulative log returns following a pattern for different future periods / candles
    :param df: the stock dataset
    :param pattern_mask: a boolean mask with True where patterns occur
    :param periods: the range of periods to calculate the returns
    :return: a dictionary with the cumulative log returns for each period. Keys are the periods ( = 1,2,3,4,...), values are lists of log returns
    """
    periods = range(1, max_ahead + 1)
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
    :return:    a dictionary with the win rate for each period. 
                Keys are the periods ( = 1,2,3,...), values are the win rate rounded to 2 decimal places
    """
    win_rate = {period: round(sum(1 for r in ret if r > 0) *100 / len(ret), 2) for period, ret in returns.items() if ret}
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
    #median_return = {period: sorted(ret)[len(ret) // 2] for period, ret in returns.items() if ret}     # valid only for odd number of elements
    median_return = {period: round(statistics.median(ret),3) for period, ret in returns.items() if ret}
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


# RESAMPLING functions:

def bootstrap_sample(returns: dict[int, list[float]], dim_sample: int) -> dict[int, list[float]]:
    """
    Generate additional samples using bootstrap method.
    :param returns: Dictionary of original returns
    :param dim_sample: Number of samples to generate
    :return: Dictionary of bootstrapped samples
    """
    bootstrapped_returns = {}
    for day, values in returns.items():
        bootstrapped_values = np.random.choice(values, size=(dim_sample,), replace=True)
        bootstrapped_returns[day] = bootstrapped_values.tolist()
    return bootstrapped_returns


def kde_sample(returns: dict[int, list[float]], dim_sample: int) -> dict[int, list[float]]:
    """
    Generate additional samples using Kernel Density Estimation suitable for leptokurtic distributions.
    
    :param returns: Dictionary of original returns
    :param dim_sample: Number of samples to generate
    :return: Dictionary of KDE samples
    """
    kde_returns = {}
    for day, values in returns.items():
        # Use a Student's t-kernel which is better suited for heavy-tailed distributions
        kde = stats.gaussian_kde(values, bw_method='scott')
        kde.set_bandwidth(kde.factor / 2.)  # Reduce bandwidth to capture more detail
        kde_samples = kde.resample(dim_sample)
        kde_returns[day] = kde_samples.flatten().tolist()
    return kde_returns

'''
def bayesian_sample(returns: dict[int, list[float]], n_samples: int) -> dict[int, list[float]]:
    """
    Generate additional samples using a Bayesian model with a mixture of Gaussians as prior.
    
    :param returns: Dictionary of original returns
    :param n_samples: Number of samples to generate
    :return: Dictionary of samples from the posterior distribution
    """
    bayesian_returns = {}
    for day, values in returns.items():
        print(day)
        # Fit a Gaussian Mixture Model to use as prior
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(np.array(values).reshape(-1, 1))
        
        # Define the Bayesian model
        with pm.Model() as model:
            # Prior
            weights = pm.Dirichlet('weights', a=np.ones(3))
            means = pm.Normal('means', mu=gmm.means_.flatten(), sigma=1, shape=3)
            stds = pm.HalfNormal('stds', sigma=1, shape=3)

            components = pm.Normal.dist(mu=means, sigma=stds, shape=(3,))
            
            pm.Mixture('likelihood',
                        w=weights,
                        comp_dists=components,
                        observed=values)
            
            # Fit the Bayesian model and obtain the posterior distribution of parameters
            #trace = pm.sample(draws=1000, tune=1500, return_inferencedata=False)    # return_inferencedata=False to avoid using with az.extract
            num_samples = 1000      # 1000=default; number of samples to generate from the posterior predictive distribution
            num_chains = 4          # number of MCMC chains. 
            posterior = pm.sample(draws=num_samples, tune=1500, chains=num_chains)
        
        # Generate new samples from the posterior predictive distribution
        with model:
            #posterior_predictive = pm.sample_posterior_predictive(trace, var_names=['likelihood'], samples=n_samples)
            pm.sample_posterior_predictive(posterior, var_names=['likelihood'], extend_inferencedata=True)
            # NOW posterior object will include many parameters and in the posterior predictive group
            # it will have generated posterior observations. they will be in 3 dimension: 
            # 4 (num_chains) x 1000 (num_samples) x len(values) (that is the sample size).

        #bayesian_returns = az.extract(posterior_predictive, group="posterior_predictive", num_samples=100)["likelihood"]
        np_object_returns = np.array(posterior.posterior_predictive['likelihood'])      # likelihood is the name of tha mixture model defined above
        concatenated_returns = np.concatenate(np_object_returns, axis=0)
        flattened_returns = concatenated_returns.flatten()

        bayesian_returns[day] = np.random.choice(flattened_returns, size=n_samples, replace=False).tolist()
    
    return bayesian_returns
'''

def bayesian_sample(returns: dict[int, list[float]], dim_sample: int, pool_size: int = None, data_weight: int = 3) -> dict[int, list[float]]:
    """
    Generate additional samples using Bayesian inference starting from a Student's t-distribution as prior.
    We overweight the original data by a factor of data_weight to give it more importance.
    If pool_size is provided, we generate a pool of samples for future use instead of returning a single sample.
    :param returns: Dictionary of original returns. Keys are days and values are lists of returns.
    :param dim_sample: Number of samples to generate from the posterior predictive distribution
    :param pool_size: Number of samples to generate and store for future use. Possible values should be around 100_000 / 150_000
    :param data_weight: Factor to overweight the original data
    :return: Dictionary of samples from the posterior distribution
    """
    bayesian_returns = {}
    bayesian_pool = {}
    for day, values in returns.items():
        print(f"Processing day {day}")

        # Original data plus bootstrapped samples
        original_data = np.array(values)
        bootstrapped_data = np.concatenate([original_data] + 
                                           [np.random.choice(original_data, size=(original_data.shape[0],), replace=True)
                                            for _ in range(data_weight - 1)])

        
        with pm.Model() as model:
            # Hyperpriors
            nu = pm.Gamma('nu', alpha=2, beta=0.1)  # Degrees of freedom
            sigma = pm.HalfCauchy('sigma', beta=1)  # Scale
            mu = pm.Normal('mu', mu=np.median(values), sigma=1)  # Location
            
            # Likelihood
            pm.StudentT('likelihood', nu=nu, mu=mu, sigma=sigma, observed=bootstrapped_data)
            
            # Sample from the posterior
            idata = pm.sample(draws=1000, tune=1500, chains=4, target_accept=0.9, return_inferencedata=True)
            
            # Generate posterior predictive samples
            pm.sample_posterior_predictive(idata, extend_inferencedata=True)
        
        # Extract posterior predictive samples
        post_pred = np.array(az.extract(idata, group="posterior_predictive")["likelihood"]).flatten()

        if pool_size:
            # Create a pool of samples
            bayesian_pool[day] = np.random.choice(post_pred, size=pool_size, replace=False).tolist()
        else:
            # Randomly select dim_sample from the posterior predictive
            bayesian_returns[day] = np.random.choice(post_pred, size=dim_sample, replace=False).tolist()
    
    if pool_size:
        return bayesian_pool
    else:
        return bayesian_returns
    

def generate_bayesian_pool(returns: dict[int, list[float]], pool_size: int = 150_000, data_weight: int = 2) -> dict[int, list[float]]:
    """
    Generate a pool of Bayesian samples for future use.
    Replicate the Bayesian sampling process multiple times and store the samples in a pool.
    :param returns: Dictionary of original returns. Keys are days and values are lists of returns.
    :param pool_size: Number of samples to generate and store for future use
    :param data_weight: Factor to overweight the original data
    :return: Dictionary of samples from the posterior distribution. keys are days and values are lists of returns
    """
    bayesian_pool = {}
    for _ in range(5):
        bayesian_returns = bayesian_sample(returns, dim_sample=None, pool_size=pool_size, data_weight=data_weight)
        for day in returns.keys():
            # First sample generated
            if day not in bayesian_pool:
                bayesian_pool[day] = bayesian_returns[day]
            # Append the new samples to the pool
            else:
                bayesian_pool[day].extend(bayesian_returns[day])
    
    return bayesian_pool


def mixed_sample(returns: dict[int, list[float]], dim_sample: int, bayesian_pool: dict[int, list[float]] = None) -> dict[int, list[float]]:
    """
    Generate samples using a mix of bootstrap, KDE, and Bayesian methods.
    In case a pool of Bayesian samples is provided, sample from it instead of generating new samples.
    
    :param returns: Dictionary of original returns
    :param dim_sample: Total number of samples to generate per day
    :return: Dictionary of samples from the mixed methods
    """
    n_bootstrap = int(0.4 * dim_sample)      # 40% of the samples
    n_kde = int(0.3 * dim_sample)            # 30% of the samples
    n_bayesian = dim_sample - (n_bootstrap + n_kde)  # Ensure the total sums up to dim_sample

    bootstrap_returns = bootstrap_sample(returns, n_bootstrap)
    kde_returns = kde_sample(returns, n_kde)
    if bayesian_pool:
        # if a pool of samples is provided, sample from it
        bayesian_returns = {}
        for day in returns.keys():
            # Sample without replacement from the Bayesian pool
            bayesian_returns[day] = np.random.choice(bayesian_pool[day], size=n_bayesian, replace=False)
    else:
        # otherwise, generate new samples
        bayesian_returns = bayesian_sample(returns, n_bayesian)

    mixed_returns = {}
    for day in returns.keys():
        mixed_returns[day] = (
            bootstrap_returns[day] + kde_returns[day] + bayesian_returns[day]
        )
    
    return mixed_returns

def generate_pattern_returns(returns: dict[int, list[float]], dim_sample: int, n_iterations: int, bayesian_pool: dict[int, np.ndarray]) -> list[dict[int, np.ndarray[float]]]:

    n_bootstrap = int(0.4 * dim_sample)  # 40% of the samples
    n_kde = int(0.3 * dim_sample)  # 30% of the samples
    n_bayesian = dim_sample - (n_bootstrap + n_kde)  # Ensure the total sums up to n_samples
    
    all_mixed_samples = []
    # Generate n_iterations mixed samples
    for i in range(n_iterations):

        bootstrap_returns = bootstrap_sample(returns, n_bootstrap)
        kde_returns = kde_sample(returns, n_kde)
        
        mixed_returns = {}
        for day in returns.keys():
            # Sample from the Bayesian pool
            bayesian_samples = np.random.choice(bayesian_pool[day], size=n_bayesian, replace=False)
            
            # Combine the samples
            mixed_returns[day] = np.array(
                bootstrap_returns[day] + 
                kde_returns[day] + 
                bayesian_samples.tolist()
            )
        
        all_mixed_samples.append(mixed_returns)
        print(f'Generated {i+1} samples.')

    return all_mixed_samples


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



def compare_distributions(baseline_results: dict, pattern_results: dict, metric: str):
    """
    Perform statistical tests comparing baseline and pattern distributions for a specific metric.
    
    :param baseline_results: Dictionary of baseline results. Keys are metrics, values are dictionaries with days as keys and lists of values as values.
    :param pattern_results: Dictionary of pattern results. Same structure as baseline_results.
    :param metric: String specifying which metric to compare (e.g., 'average_return')
    """
    print(f"Statistical tests for {metric}:")
    print("=" * 50)

    for day in baseline_results[metric].keys():
        baseline_data = np.array(baseline_results[metric][day])
        pattern_data = np.array(pattern_results[metric][day])

        # Calculate differences in mean and median
        mean_diff = np.mean(pattern_data) - np.mean(baseline_data)
        median_diff = np.median(pattern_data) - np.median(baseline_data)

        # Perform t-test for means
        t_stat, t_pvalue = stats.ttest_ind(pattern_data, baseline_data)

        # Perform Mann-Whitney U test for medians
        u_stat, u_pvalue = stats.mannwhitneyu(pattern_data, baseline_data, alternative='two-sided')

        # Perform Kolmogorov-Smirnov test for distribution shape
        ks_stat, ks_pvalue = stats.ks_2samp(pattern_data, baseline_data)

        # Perform Levene test for equality of variances
        levene_stat, levene_pvalue = stats.levene(pattern_data, baseline_data)

        print(f"\nDay {day}:")
        print(f"  Mean difference (Pattern - Baseline): {mean_diff:.4f}")
        print(f"  Median difference (Pattern - Baseline): {median_diff:.4f}")
        print(f"  T-test (means): p-value = {t_pvalue:.4f}")
        print(f"  Mann-Whitney U test (medians): p-value = {u_pvalue:.4f}")
        print(f"  Kolmogorov-Smirnov test (distribution shape): p-value = {ks_pvalue:.4f}")
        print(f"  Levene test (equality of variances): p-value = {levene_pvalue:.4f}")