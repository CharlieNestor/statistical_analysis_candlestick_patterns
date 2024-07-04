import pandas as pd


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
        (body > 2 * prev_body)
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
        (body > 2 * prev_body)
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
        bullish_harami
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


# Dictionary of patterns
patterns = {
    'Bullish Engulfing': {'function': bullish_engulfing, 'candles': 2, 'direction': 1},
    'Bearish Engulfing': {'function': bearish_engulfing, 'candles': 2, 'direction': -1},
    'Bullish Harami': {'function': bullish_harami, 'candles': 2, 'direction': 1},
    'Bearish Harami': {'function': bearish_harami, 'candles': 2, 'direction': -1},
}