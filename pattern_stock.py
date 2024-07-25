import numpy as np
import pandas as pd
import patterns as pt
import analysis as an
from loader import load_data, check_clean_data


class PatternStock:
    """
    A class to represent a stock and perform candlestick pattern analysis on it.
    This class handles loading stock data, applying candlestick patterns,
    calculating metrics, and running simulations for statistical analysis.
    """

    def __init__(self, ticker):
        """
        Initialize the PatternStock object.
        :param ticker: The stock ticker symbol
        """
        self.ticker = ticker
        self.df = None          # Main DataFrame to store historical price data
        self.info = None        # Dictionary to store general stock information
        self.support_df = None  # DataFrame to store supporting indicators (e.g., ATR)
        self.pattern_data = {}  # Dictionary to store pattern-specific data and analysis results


    def load_data(self):
        """
        Load and preprocess the stock data.
        This method fetches the stock data, cleans it, adds percentage and log returns,
        and calculates the Average True Range (ATR) indicator.
        """
        data = load_data(self.ticker)
        if not data or self.ticker not in data:
            raise ValueError(f"Invalid ticker or no data found for {self.ticker}")
        cleaned_data = check_clean_data(data)
        if not cleaned_data[self.ticker]['historical_data'].empty:
            self.df = cleaned_data[self.ticker]['historical_data']
            self.info = cleaned_data[self.ticker]['info']
            self.df = an.add_pct_log_returns(self.df)
            self.support_df = an.calculate_ATR(self.df)
        else:
            raise ValueError(f"No historical data available for {self.ticker}")


    def apply_pattern(self, pattern_name):
        """
        Apply a specific candlestick pattern to the stock data.
        This method identifies the occurrences of the specified pattern in the stock's price history.
        :param pattern_name: The name of the candlestick pattern to apply
        """
        if pattern_name not in self.pattern_data:
            pattern_info = pt.patterns[pattern_name]
            pattern_function = pattern_info['function']
            # Combine main DataFrame and support DataFrame for pattern detection
            mask = pattern_function(pd.concat([self.df, self.support_df], axis=1))
            
            self.pattern_data[pattern_name] = {
                'info': pattern_info,
                'mask': mask,
                'dim_pattern': mask.sum(),      # Number of pattern occurrences
                'metrics': None,
                'simulation_results': None
            }
        
        self.calculate_metrics(pattern_name)


    def calculate_metrics(self, pattern_name):
        """
        Calculate performance metrics for the specified pattern.
        This method computes win rate, average return, median return, and standard deviation
        of returns following each occurrence of the pattern.
        :param pattern_name: The name of the candlestick pattern
        """
        if self.pattern_data[pattern_name]['metrics'] is None:
            mask = self.pattern_data[pattern_name]['mask']
            returns = an.calculate_log_cumReturns_periods(self.df, mask, max_ahead=15)
            
            self.pattern_data[pattern_name]['metrics'] = {
                'win_rate': an.calculate_win_rate_from_log(returns),
                'average_return': an.calculate_average_return_from_log(returns),
                'median_return': an.calculate_median_return_from_log(returns),
                'std_return': an.calculate_std_return_from_log(returns),
            }

    
    def run_simulation(self, pattern_name, n_iterations=1000):
        """
        Run n_interations simulations to assess the statistical significance of the pattern.
        This method generates random samples, calculates confidence intervals and p-values,
        and creates a significance table comparing the pattern's performance to random chance.
        :param pattern_name: The name of the candlestick pattern
        :param n_iterations: Number of iterations for the Monte Carlo simulation (default: 1000)
        """
        if self.pattern_data[pattern_name]['simulation_results'] is None:
            mask = self.pattern_data[pattern_name]['mask']
            dim_pattern = self.pattern_data[pattern_name]['dim_pattern']
            
            # Generate random samples and calculate their metrics
            base_returns = an.generate_random_returns(self.df, mask, dim_pattern, n_iterations, is_log=True, verbose=False)
            base_distributions = an.calculate_metrics(base_returns)
            # Calculate confidence intervals
            nonPar_conf_int = an.calculate_nonParametric_confidence_intervals(base_distributions)
            parametric_conf_int = an.calculate_parametric_confidence_intervals(base_distributions)
            
            metrics = self.pattern_data[pattern_name]['metrics']
            empirical_pvalues = an.calculate_empirical_pvalues(metrics, base_distributions)
            
            # Transform confidence intervals for easier comparison
            param_conf_int, basecase_estimates = an.transform_confidence_intervals(parametric_conf_int)
            non_parametric_conf_int, _ = an.transform_confidence_intervals(nonPar_conf_int)
            
            # Create a table showing the statistical significance of the pattern's performance
            significance_table = an.create_significance_table(
                metrics, basecase_estimates, empirical_pvalues, param_conf_int, non_parametric_conf_int
            )
            
            self.pattern_data[pattern_name]['simulation_results'] = {
                'base_distributions': base_distributions,
                'nonPar_conf_int': nonPar_conf_int,
                'parametric_conf_int': parametric_conf_int,
                'empirical_pvalues': empirical_pvalues,
                'significance_table': significance_table
            }

    def get_data_for_plotting(self, pattern_name):
        """
        Retrieve all necessary data for plotting the pattern analysis results.
        This method collects the relevant data for creating visualizations of the pattern analysis,
        including the stock data, pattern occurrences, metrics, and simulation results.
        :param pattern_name: The name of the candlestick pattern
        :return: A dictionary containing all the data needed for plotting, or None if the pattern hasn't been analyzed
        """
        if pattern_name not in self.pattern_data:
            return None
        
        pattern_data = self.pattern_data[pattern_name]
        
        data = {
            'df': self.df,
            'mask': pattern_data['mask'],
            'pattern_name': pattern_name,
            'pattern_info': pattern_data['info'],
            'pattern_metrics': pattern_data['metrics'],
        }
        
        if pattern_data['simulation_results']:
            data['confidence_intervals'] = pattern_data['simulation_results']['nonPar_conf_int']
            data['par_confidence_intervals'] = pattern_data['simulation_results']['parametric_conf_int']
            data['significance_table'] = pattern_data['simulation_results']['significance_table']
        
        return data
