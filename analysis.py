import statistics
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture


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
        new_df.iloc[i].at['ATR'] = (new_df.iloc[i-1].at['ATR'] * (period - 1) + new_df.iloc[i].at['TR']) / period
        #new_df.at[i, 'ATR'] = (new_df.at[i-1, 'ATR'] * (period - 1) + new_df.at[i, 'TR']) / period

    # Drop the intermediate 'Prev_Close' column
    df.drop(columns=['Prev_Close'], inplace=True)

    return new_df

# ANALYSIS functions relative to each periods (1 to 10 days) and metrics

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

# RESAMPLING functions:

def bootstrap_sample(returns: dict[int, list[float]], n_samples: int) -> dict[int, list[float]]:
    """
    Generate additional samples using bootstrap method.
    :param returns: Dictionary of original returns
    :param n_samples: Number of samples to generate
    :return: Dictionary of bootstrapped samples
    """
    bootstrapped_returns = {}
    for day, values in returns.items():
        bootstrapped_values = np.random.choice(values, size=(n_samples,), replace=True)
        bootstrapped_returns[day] = bootstrapped_values.tolist()
    return bootstrapped_returns


def kde_sample(returns: dict[int, list[float]], n_samples: int) -> dict[int, list[float]]:
    """
    Generate additional samples using Kernel Density Estimation suitable for leptokurtic distributions.
    
    :param returns: Dictionary of original returns
    :param n_samples: Number of samples to generate
    :return: Dictionary of KDE samples
    """
    kde_returns = {}
    for day, values in returns.items():
        # Use a Student's t-kernel which is better suited for heavy-tailed distributions
        kde = stats.gaussian_kde(values, bw_method='scott')
        kde.set_bandwidth(kde.factor / 2.)  # Reduce bandwidth to capture more detail
        kde_samples = kde.resample(n_samples)
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

def bayesian_sample(returns: dict[int, list[float]], n_samples: int, data_weight: int = 2) -> dict[int, list[float]]:
    bayesian_returns = {}
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
        
        # Randomly select n_samples from the posterior predictive
        bayesian_returns[day] = np.random.choice(post_pred, size=n_samples, replace=False).tolist()
    
    return bayesian_returns

def mixed_sample(returns: dict[int, list[float]], n_samples: int) -> dict[int, list[float]]:
    """
    Generate samples using a mix of bootstrap, KDE, and Bayesian methods.
    
    :param returns: Dictionary of original returns
    :param n_samples: Total number of samples to generate per day
    :return: Dictionary of samples from the mixed methods
    """
    n_bootstrap = int(0.4 * n_samples)
    n_kde = int(0.3 * n_samples)
    n_bayesian = n_samples - (n_bootstrap + n_kde)  # Ensure the total sums up to n_samples

    bootstrap_returns = bootstrap_sample(returns, n_bootstrap)
    kde_returns = kde_sample(returns, n_kde)
    bayesian_returns = bayesian_sample(returns, n_bayesian)

    mixed_returns = {}
    for day in returns.keys():
        mixed_returns[day] = (
            bootstrap_returns[day] + kde_returns[day] + bayesian_returns[day]
        )
    
    return mixed_returns