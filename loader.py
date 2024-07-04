import pandas as pd
import yfinance as yf
from typing import Union


def load_data(tickers: Union[str, list[str]], start_date = '1995-01-01') -> dict[str, dict[str, any]]:
    """
    Load stock data from Yahoo Finance starting from 1995-01-01.
    :param tickers: set of tickers to load
    :param start_date: start date for historical data
    :return: dictionary with stock data. keys are tickers and values are dictionaries with keys 'info', 'historical_data', 'splits'
    """
    stock_data = {}
    # if tickers is a string, convert it to a list
    if isinstance(tickers, str):
        tickers = [tickers]
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        # get stock info
        try:
            stock_info = stock.info
        except Exception as e:
            print(f"Could not get info for {ticker}")
            print(e)
            stock_info = None
            continue    # skip to the next ticker
        # get historical data
        try:
            historical_data = stock.history(start=start_date)     # keepna=True will keep the rows with missing values
        except Exception as e:
            print(f"Could not get historical data for {ticker}")
            print(e)
            historical_data = None
            continue
        # get stock splits
        try:
            splits = stock.splits
        except Exception as e:
            print(f"Could not get splits for {ticker}")
            print(e)
            splits = None
            continue
        # store the data in the dictionary
        stock_data[ticker] = {
            'info': stock_info,
            'historical_data': historical_data,
            'splits': splits,
        }

    return stock_data


def round_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Round the values in the dataframe
    """
    columns_to_round = ['Open', 'High', 'Low', 'Close']
    # round the values in the dataframe depending on the stock prices
    # price > 10: round to 2 decimal places, price > 1: round to 3 decimal places, price < 1: round to 4 decimal places
    for column in columns_to_round:
        df[column] = df[column].apply(lambda x: round(x, 2) if x > 10 else round(x, 3) if x > 1 else round(x, 4))
    return df


def remove_typos_and_missing_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Remove rows with missing or incorrect data. Missing values, negative prices, and incorrect Open, High, Low, Close values are removed.
    :param df: dataframe with historical data
    """
    initial_rows = len(df)
    # remove rows with missing values
    #df = df.dropna(how='all')       # to drop if all value in the row are NaN
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'], how='any')

    # remove rows with negative prices (valid for stocks)
    df = df[(df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)]

    # remove rows with Open, High, Low, Close values that don't make sense
    df = df[df['High'] >= df['Low']]
    df = df[df['High'] >= df['Open']]
    df = df[df['High'] >= df['Close']]
    df = df[df['Low'] <= df['Open']]
    df = df[df['Low'] <= df['Close']]

    final_rows = len(df)
    if initial_rows != final_rows:
        print(f'From {ticker} dataset were removed {initial_rows - final_rows} rows with missing or incorrect data')

    return df


def check_prices_volumes_excursion(df: pd.DataFrame) -> Union[list[str], float]:
    """
    Check for dates with same OHLC prices, low volume, and calculate the average price excursion
    :param df: dataframe with historical data
    :return: list of dates with same prices, list of dates with low volume, average price excursion
    """
    # Dates with OHLC all the same
    same_price_dates = df[(df['Open'] == df['High']) & (df['High'] == df['Low']) & (df['Low'] == df['Close'])].index.tolist()
    
    # Dates with low volume
    low_volume_dates = df[df['Volume'] < 1000].index.tolist()
    
    # Calculate the average price
    avgPrice = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    
    # Calculate the excursion of the average price
    max_avg_price = avgPrice.max()
    min_avg_price = avgPrice.min()
    excursion = (max_avg_price - min_avg_price) / min_avg_price * 100
    
    return same_price_dates, low_volume_dates, excursion


def identify_anomalies(df: pd.DataFrame, threshold1: float = 0.35, threshold2: float = 0.50) -> dict[str, list]:
    """
    Identify anomalies in the stock data: cases where variations between prices are too high
    :param df: dataframe with historical data
    :param threshold1: threshold for the percentage difference between open and close prices
    :param threshold2: threshold for the percentage difference between high and low prices
    :return: dictionary with anomalies. Keys are 'Open-pClose Anomalies', 'High-Low Anomalies', 'Close-Open Anomalies'
    """
    anomalies = {
        'Open-pClose Anomalies': [],
        'High-Low Anomalies': [],
        'Close-Open Anomalies': []
    }

    for i in range(1, len(df)):
        previous_close = df.iloc[i-1]['Close']
        current_open = df.iloc[i]['Open']
        current_high = df.iloc[i]['High']
        current_low = df.iloc[i]['Low']
        current_close = df.iloc[i]['Close']

        # check if the open is more than 35% higher or lower than the previous close
        if abs(current_open - previous_close) / previous_close > threshold1:
            anomalies['Open-pClose Anomalies'].append((df.index[i], current_open, previous_close))

        # check if the high-low excursion is more than 50%
        if (current_high - current_low) / current_low > threshold2:
            anomalies['High-Low Anomalies'].append((df.index[i], current_high, current_low))

        # check if the close is more than 35% higher or lower than the open
        if abs(current_close - current_open) / current_open > threshold1:
            anomalies['Close-Open Anomalies'].append((df.index[i], current_close, current_open))

    return anomalies


def check_clean_data(stock_data: dict[str, dict[str, any]]) -> dict[str, dict[str, any]]:

    """
    Check and clean the stock data and print warn for potential issues in the dataset.
    :param stock_data: dictionary with stock data
    :return: dictionary with cleaned stock data. keys are tickers and values are dictionaries with keys 'info', 'historical_data', 'splits'
    """

    for ticker, data in stock_data.items():
        historical_data = data.get('historical_data')
        if historical_data is not None and not historical_data.empty:

            # remove typos and missing data
            cleaned_data = remove_typos_and_missing_data(historical_data, ticker)

            # round the values
            rounded_data = round_values(cleaned_data)
            stock_data[ticker]['historical_data'] = rounded_data
            
            # check prices and average excursion
            same_price_dates, low_volume_dates, avg_excursion = check_prices_volumes_excursion(rounded_data)

            # identify anomalies
            anomalies = identify_anomalies(rounded_data)
            num_anomalies = sum([len(anomalies[key]) for key in anomalies.keys()])
            
            # output the results
            if same_price_dates:
                print(f"Ticker: {ticker} has the same OHLC prices on {len(same_price_dates)} dates")
                #print(same_price_dates)
            if low_volume_dates:
                print(f"Ticker: {ticker} has low volume on {len(low_volume_dates)} dates")
                #print(low_volume_dates)
            if avg_excursion < 75:
                print(f"Ticker: {ticker} has an average price excursion of less than 75%: {avg_excursion:.2f}%")
            if num_anomalies:
                print(f"Ticker: {ticker} has {num_anomalies} anomalies:")
                for key, value in anomalies.items():
                    if value:
                        print(f"  {key}: {len(value)}")
                        #print(value)

    return stock_data

