import random
import numpy as np
import pandas as pd


def three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """
    Three White Soldiers pattern: three consecutive bullish candles with higher closes and opens.
    Filters: the body of each candle should be at least 50% of the current ATR.
    """

    body = abs(df['Close'] - df['Open'])
    full_range = abs(df['High'] - df['Low'])
    direction = df['Close'] > df['Open']

    # Shifted columns for the previous candle and the candle before that
    prev_open = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)
    prev_body = body.shift(1)
    prev_direction = direction.shift(1)
    second_prev_open = df['Open'].shift(2)
    second_prev_close = df['Close'].shift(2)
    second_prev_body = body.shift(2)
    second_prev_direction = direction.shift(2)

    # Indicators shifted
    prev_atr = df['ATR'].shift(1)
    second_prev_atr = df['ATR'].shift(2)

    # Q: Do you want to specify that close should be near the high of the candle?
    # Three white soldiers pattern conditions
    three_white_soldiers = (
        (direction == True) &                   # Current candle is bullish
        (prev_direction == True) &              # Previous candle was bullish
        (second_prev_direction == True) &       # Second previous candle was bullish
        (df['Open'] > prev_open) &              # Current open is lower than previous close (no gap up)
        (df['Close'] > prev_close) &            # Current close is higher than previous close
        (prev_open > second_prev_open) &        # Previous open is lower than second previous close (no gap up)
        (prev_close > second_prev_close)        # Previous close is higher than second previous close
    )

    soldiers_mask = (
        three_white_soldiers &
        (body > 0.5 * df['ATR']) &
        (prev_body > 0.5 * prev_atr) &
        (second_prev_body > 0.5 * second_prev_atr)
        # any further conditions or filters can be added here ...
    )
    return soldiers_mask


def three_black_crows(df: pd.DataFrame) -> pd.Series:
    """
    Three Black Crows pattern: three consecutive bearish candles with lower closes and opens.
    Filters: the body of each candle should be at least 50% of the current ATR.
    """

    body = abs(df['Close'] - df['Open'])
    full_range = abs(df['High'] - df['Low'])
    direction = df['Close'] > df['Open']

    # Shifted columns for the previous candle
    prev_open = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)
    prev_body = body.shift(1)
    prev_direction = direction.shift(1)
    second_prev_open = df['Open'].shift(2)
    second_prev_close = df['Close'].shift(2)
    second_prev_body = body.shift(2)
    second_prev_direction = direction.shift(2)

    # Three black crows pattern conditions
    three_black_crows = (
        (direction == False) &                  # Current candle is bearish
        (prev_direction == False) &             # Previous candle was bearish
        (second_prev_direction == False) &      # Second previous candle was bearish
        (df['Open'] < prev_open) &              # Current open is higher than previous close (no gap down)
        (df['Close'] < prev_close) &            # Current close is lower than previous close
        (prev_open < second_prev_open) &        # Previous open is higher than second previous close (no gap down)
        (prev_close < second_prev_close)        # Previous close is lower than second previous close
    )

    crows_mask = (
        three_black_crows &
        (body > 0.35 * df['ATR']) &
        (prev_body > 0.35 * df['ATR']) &
        (second_prev_body > 0.35 * df['ATR'])
        # any further conditions or filters can be added here ...
    )
    return crows_mask


def morning_star(df: pd.DataFrame) -> pd.Series:
    """
    Morning Star pattern: a bearish candle followed by continuation is countered by a gap up bullish candle that closes near the high.
    Filters: the body of the second candle should be at least 50% of the current ATR.
    """
    # Calculate body - range - direction of each candle
    body = abs(df['Close'] - df['Open'])
    full_range = abs(df['High'] - df['Low'])
    direction = df['Close'] > df['Open']
    midpoint = df['Open'] + (df['Close'] - df['Open'])/2

    # Shifted columns for the previous candle
    prev_open = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)
    prev_high = df['High'].shift(1)
    prev_low = df['Low'].shift(1)
    prev_body = body.shift(1)
    prev_direction = direction.shift(1)
    second_prev_open = df['Open'].shift(2)
    second_prev_close = df['Close'].shift(2)
    second_prev_body = body.shift(2)
    second_prev_direction = direction.shift(2)
    second_prev_midpoint = midpoint.shift(2)

    # Morning star pattern conditions
    morning_star = (
        (direction == True) &                   # Current candle is bullish
        (second_prev_direction == False) &      # Previous candle was bearish
        (df['Open'] > prev_close) &             # Gap Up (considering close only)
        (df['Close'] > second_prev_midpoint) &             # Current close is higher than the midpoint of the candle 2 days ago
        (df['Low'] > prev_low) &                # Current low is higher than previous low
        (prev_open < second_prev_close) &       # Gap Down (considering close only)
        (prev_close < second_prev_close)        # Previous close is lower than the close 2 days ago
    )

    star_mask = (
        morning_star &
        # any further conditions or filters can be added here ...
        (second_prev_body > 0.5 * df['ATR']) &
        (body > 0.5 * df['ATR'])
        # like middle candle creates a true gap
    )
    return star_mask


def evening_star(df: pd.DataFrame) -> pd.Series:
    """
    Evening Star pattern: a bullish candle followed by continuation is countered by a gap down bearish candle that closes near the low.
    """
    # Calculate body - range - direction of each candle
    body = abs(df['Close'] - df['Open'])
    full_range = abs(df['High'] - df['Low'])
    direction = df['Close'] > df['Open']
    midpoint = df['Open'] + (df['Close'] - df['Open'])/2

    # Shifted columns for the previous candle
    prev_open = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)
    prev_high = df['High'].shift(1)
    prev_low = df['Low'].shift(1)
    prev_body = body.shift(1)
    prev_direction = direction.shift(1)
    second_prev_open = df['Open'].shift(2)
    second_prev_close = df['Close'].shift(2)
    second_prev_body = body.shift(2)
    second_prev_direction = direction.shift(2)
    second_prev_midpoint = midpoint.shift(2)

    # Evening star pattern conditions
    evening_star = (
        (direction == False) &                  # Current candle is bearish
        (second_prev_direction == True) &       # Previous candle was bullish
        (df['Open'] < prev_close) &             # Gap Down (considering close only)
        (df['Close'] < second_prev_midpoint ) &             # Current close is lower than the midpoint of the candle 2 days ago
        (df['High'] < prev_high) &              # Current high is lower than previous high
        (prev_open > second_prev_close) &       # Gap Up (considering close only)
        (prev_close > second_prev_close)        # Previous close is higher than the close 2 days ago
    )

    star_mask = (
        evening_star &
        # any further conditions or filters can be added here ...
        (second_prev_body > 0.5 * df['ATR']) &
        (body > 0.5 * df['ATR'])
        # like middle candle creates a true gap
    )
    return star_mask


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Bullish Engulfing pattern: the second candle is bullish and its body engulfs the body of the previous candle.
    Filters: the body of the second candle > 2 * body of the first candle.
    """
    # calculate the body size and range of each candle
    body = abs(df['Close'] - df['Open'])
    full_range = abs(df['High'] - df['Low'])
    # determine if the candle is bullish (True) or bearish (False)
    direction = df['Close'] > df['Open']

    # shift columns to keep track of the previous candle
    prev_open = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)
    prev_body = body.shift(1)
    prev_direction = direction.shift(1)

    # bullish engulfing pattern conditions
    bullish_engulfing = (
        (direction == True) &           # current candle is bullish
        (prev_direction == False) &     # previous candle was bearish
        (df['Close'] > prev_open) &     # current close is higher than previous open
        (df['Open'] < prev_close)       # current open is lower than previous close
    )

    engulfing_mask = (
        bullish_engulfing &
        (body > 2 * prev_body) &
        (full_range >= df['ATR'])
        # any further conditions or filters can be added here ...
        # ATR ...
    )
    return engulfing_mask


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Bearish Engulfing pattern: the second candle is bearish and its body engulfs the body of the previous candle.
    Filters: the body of the second candle > 2 * body of the first candle.
    """
    # calculate the body size and range of each candle
    body = abs(df['Close'] - df['Open'])
    full_range = abs(df['High'] - df['Low'])
    # determine if the candle is bullish (True) or bearish (False)
    direction = df['Close'] > df['Open']

    # shift columns to keep track of the previous candle
    prev_open = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)
    prev_body = body.shift(1)
    prev_direction = direction.shift(1)

    # bearish engulfing pattern conditions
    bearish_engulfing = (
        (direction == False) &          # current candle is bearish
        (prev_direction == True) &      # previous candle was bullish
        (df['Open'] > prev_close) &     # current open is higher than previous close
        (df['Close'] < prev_open)       # current close is lower than previous open
    )

    engulfing_mask = (
        bearish_engulfing &
        (body > 2 * prev_body) &
        (full_range >= df['ATR'])
        # any further conditions or filters can be added here ...
        # ATR ...
    )
    return engulfing_mask


def bullish_harami(df: pd.DataFrame) -> pd.Series:
    """
    Bullish Harami pattern: the previous candle is (very) bearish, the current is completely inside the previous candle. Reversal pattern.
    """
    # calculate body - range - direction of each candle
    body = abs(df['Close'] - df['Open'])
    full_range = abs(df['High'] - df['Low'])
    direction = df['Close'] > df['Open']

    # keep track of the previous candle
    prev_open = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)
    prev_high = df['High'].shift(1)
    prev_low = df['Low'].shift(1)
    prev_body = body.shift(1)
    prev_range = full_range.shift(1)
    prev_direction = direction.shift(1)

    # bullish harami pattern conditions
    bullish_harami = (
        (direction == True) &           # current candle is bullish
        (prev_direction == False) &     # previous candle was bearish
        (df['Close'] < prev_open) &     # current close is lower than previous open
        (df['Open'] > prev_close) &     # current open is higher than previous close
        (df['High'] < prev_high) &      # current high is lower than previous high
        (df['Low'] > prev_low)          # current low is higher than previous low
    )

    harami_mask = (
        bullish_harami & 
        (prev_range >= df['ATR'])
        # any further conditions or filters can be added here ...
        # like previous candle has a large body
    )
    return harami_mask

def bearish_harami(df: pd.DataFrame) -> pd.Series:
    """
    Bearish Harami pattern: the previous candle is (very) bullish, the current is completely inside the previous candle. Reversal pattern.
    """
    # calculate body - range - direction of each candle
    body = abs(df['Close'] - df['Open'])
    full_range = abs(df['High'] - df['Low'])
    direction = df['Close'] > df['Open']

    # keep track of the previous candle
    prev_open = df['Open'].shift(1)
    prev_close = df['Close'].shift(1)
    prev_high = df['High'].shift(1)
    prev_low = df['Low'].shift(1)
    prev_body = body.shift(1)
    prev_direction = direction.shift(1)

    # bearish harami pattern conditions
    bearish_harami = (
        (direction == False) &          # current candle is bearish
        (prev_direction == True) &      # previous candle was bullish
        (df['Close'] > prev_open) &     # current close is higher than previous open
        (df['Open'] < prev_close) &     # current open is lower than previous close
        (df['High'] < prev_high) &      # current high is lower than previous high
        (df['Low'] > prev_low)          # current low is higher than previous low
    )

    harami_mask = (
        bearish_harami
        # any further conditions or filters can be added here ...
        # like previous candle has a large body
    )
    return harami_mask

def bullish_marubozu(df: pd.DataFrame) -> pd.Series:
    """
    Bullish Marubozu pattern
    """
    # Calculate body - range - direction of each candle
    body = abs(df['Close'] - df['Open'])
    full_range = abs(df['High'] - df['Low'])
    direction = df['Close'] > df['Open']

    # Bullish Marubozu pattern conditions
    bullish_marubozu = (
        (direction == True) &                   # Current candle is bullish
        (body/full_range > 0.9)                 # Body is at least 90% of the range
    )

    marubozu_mask = (
        bullish_marubozu &
        (body > df['ATR'])
        # any further conditions or filters can be added here ...
    )
    return marubozu_mask

def bearish_marubozu(df: pd.DataFrame) -> pd.Series:
    """
    Bearish Marubozu pattern
    """
    # Calculate body - range - direction of each candle
    body = abs(df['Close'] - df['Open'])
    full_range = abs(df['High'] - df['Low'])
    direction = df['Close'] > df['Open']

    # Bearish Marubozu pattern conditions
    bearish_marubozu = (
        (direction == False) &                  # Current candle is bearish
        (body/full_range > 0.9)                 # Body is at least 90% of the range
    )

    marubozu_mask = (
        bearish_marubozu &
        (body > df['ATR'])
        # any further conditions or filters can be added here ...
    )
    return marubozu_mask


# ARTIFICIAL MASKS

def random_mask(df: pd.DataFrame, dim_sample: int = 100, lag: int = 10) -> pd.Series:
    """
    Generate a random mask with dim_sample valid (True) size that are at least 'lag' days apart.
    """
    if dim_sample >= 0.975*len(df):
        raise ValueError("The sample size is too large compared to the dataset size.")
    elif dim_sample >= 0.8*len(df):
        print("The sample size is covering a large portion of the dataset. Consider reducing the sample size.")
        
    n = len(df)
    all_indices = list(range(1,n))      # avoid index 0 for Nan values
    valid_samples = []
    # sample without caring whether they are at least 'lag' days apart (could sample adjacent days)
    valid_samples = random.sample(all_indices, dim_sample)

    """
    attempt_count = 0  # Track the number of attempts to find valid samples
    
    while len(valid_samples) < dim_sample + 1:
        attempt_count += 1
        if attempt_count > 10000:  # If too many attempts are made, reduce lag
            lag = max(3, lag - 1)
            attempt_count = 0   # Reset the attempt counter
            valid_samples = []  # Reset the valid samples list
            print(f"Reducing lag to {lag} to find valid samples.")

        sample = random.choice(all_indices)
        if all(abs(sample - s) >= lag for s in valid_samples):  # check if the new sample is at least 'lag' days apart from any other sample
            valid_samples.append(sample)
    """
    
    neutral_mask = pd.Series([False] * n)
    neutral_mask.iloc[valid_samples] = True
    return neutral_mask


def filter_mask(mask: pd.Series, step: int) -> pd.Series:
    """
    Ensure True values in the mask are separated by at least 'step' Falses.
    :param mask: A pd.Series of boolean values.
    :param step: Minimum number of False values between True values.
    :return: A pd.Series with adjusted True values according to the step separation.
    """
    # convert mask to a numpy array
    mask_array = mask.to_numpy()
    true_indices = np.where(mask_array)[0]

    i = 0
    while i < len(true_indices):
        # find the window of True values
        window_start = true_indices[i]
        window_end = min(window_start + step + 1, len(mask_array))
        window_indices = true_indices[(true_indices >= window_start) & (true_indices < window_end)]
        
        # if there are multiple True values within the step window, randomly keep only one
        if len(window_indices) > 1:
            chosen_index = np.random.choice(window_indices)
            mask_array[window_indices] = False
            mask_array[chosen_index] = True
        
        # move to the next window
        i += len(window_indices)
    
    # convert back to pd.Series
    return pd.Series(mask_array, index=mask.index)


# dictionary of patterns
patterns = {
    'Morning Star': {'function': morning_star, 'candles': 3, 'direction': 1},
    'Evening Star': {'function': evening_star, 'candles': 3, 'direction': -1},
    'Three White Soldiers': {'function': three_white_soldiers, 'candles': 3, 'direction': 1},
    'Three Black Crows': {'function': three_black_crows, 'candles': 3, 'direction': -1},
    'Bullish Engulfing': {'function': bullish_engulfing, 'candles': 2, 'direction': 1},
    'Bearish Engulfing': {'function': bearish_engulfing, 'candles': 2, 'direction': -1},
    'Bullish Harami': {'function': bullish_harami, 'candles': 2, 'direction': 1},
    'Bearish Harami': {'function': bearish_harami, 'candles': 2, 'direction': -1},
    'Bullish Marubozu': {'function': bullish_marubozu, 'candles': 1, 'direction': 1},
    'Bearish Marubozu': {'function': bearish_marubozu, 'candles': 1, 'direction': -1},
}