import random
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_chart(dataset: pd.DataFrame, ticker: str) -> None:
    """
    Plot a candlestick chart of the stock price for the whole dataset.
    :param dataset: dataframe with historical data
    :param ticker: stock ticker
    """
    # create a candlestick chart
    fig = go.Figure( data = [go.Candlestick(x = dataset.index,
                        open  = dataset["Open"],
                        high  = dataset["High"],
                        low   = dataset["Low"],
                        close = dataset["Close"])],
                        )
    # add title, axis labels, remove rangeslider
    fig.update_layout( title = dict(text = f'{ticker} Price from 1995 onwards',
                                    font = dict(size=18, color='red'),
                                    x = 0.5,
                                    ),
                            yaxis_title = "Price (log)",
                            xaxis_title = "Date",
                            xaxis_rangeslider_visible = False,
                            autosize = True,
                            hovermode = 'x unified',
                            )
    
    fig.update_xaxes(showline = True, 
                    linewidth = 2, 
                    linecolor = 'black',
                    mirror = True,
                    tickangle = -45,
                    )

    fig.update_yaxes(type = "log",            # this adds a logarithmic scale on the y axis
                    showgrid = True,
                    gridcolor = 'blue', griddash = "longdash",
                    showline = True, linewidth = 2, 
                    linecolor = 'black', mirror = True,
                    )

    fig.show()


def plot_close_with_patterns(data: pd.DataFrame, ticker: str, mask: pd.Series) -> None:
    """
    Plot the close price of the stock with vertical lines marking the dates where patterns occur
    :param data: the stock data
    :param ticker: the stock ticker
    :param mask: a boolean mask with True where patterns occur
    """
    fig = go.Figure()
    
    # add the close price line
    fig.add_trace(go.Scatter(
        x = data.index,
        y = data['Close'],
        mode = 'lines',
        name = 'Close Price'
    ))

    # find the dates where patterns occur
    pattern_dates = data.index[mask]

    # add vertical lines and labels for each pattern
    for date in pattern_dates:
        fig.add_shape(
            type = "line",
            x0 = date, x1 = date,
            y0 = 0, y1 = 1,
            yref = "paper",           # y-coordinate is in paper coordinates, that is normalized in [0, 1]
            line = dict(color="red", width=0.5, dash="dash"),
        )
        
        # add a label with just the date (not time)
        fig.add_annotation(
            x = date - pd.Timedelta(hours=60),
            y = 0.05,
            yref = "paper",
            text = date.strftime('%Y-%m-%d'),
            showarrow = False,
            textangle = -90,
            font = dict(size=7, color='red'),
        )

    fig.update_layout(
        title = dict(text = f'{ticker} Close Price with marked patterns',
                            font=dict(size=18, color='red'),
                            x=0.5,
                            ),
        xaxis_title = 'Date',
        yaxis_title = 'Close Price (log)',
        hovermode = 'x',
        autosize = True,
    )

    fig.update_yaxes(type='log')

    fig.show()


def plot_patterns(data: pd.DataFrame, mask: pd.Series, num_candles: int, ticker: str, pattern_name: str, max_candles=20, back_candles=5, max_subplots=12) -> None:
    """
    create a subplot candlestick chart for each pattern detected up to max_subplots, randomly selected
    :param data: the stock data
    :param mask: a boolean mask with True where patterns occur
    :param candle_pattern: the number of candles in the pattern
    :param ticker: the stock ticker
    """
    # find the dates where patterns occur
    pattern_dates = data.index[mask]
    
    # select random dates from the pattern_dates up to k instances
    section_dates = random.sample(list(pattern_dates), k=max_subplots)
    section_dates = sorted(section_dates)
    
    # determine the number of rows and columns for the subplots
    n_plots = len(section_dates)
    n_cols = min(3, n_plots)        # Max 3 columns
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # create subplot figure
    fig = make_subplots(rows=n_rows, cols=n_cols, 
                        vertical_spacing=0.1, horizontal_spacing=0.05)
    
    for i, date in enumerate(section_dates):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        # get the index of the engulfing pattern
        idx = data.index.get_loc(date)
        
        # select data for the subplot. Must consider the back_candles and the num_candles
        start_idx = max(0, idx - back_candles)
        end_idx = min(len(data), start_idx + max_candles)
        subset = data.iloc[start_idx:end_idx]
        
        # create candlestick trace
        candlestick = go.Candlestick(
            x=subset.index,
            open=subset['Open'],
            high=subset['High'],
            low=subset['Low'],
            close=subset['Close'],
            showlegend=False,
            name = "",          # this is to avoid any trace number in the legend
        )
        
        fig.add_trace(candlestick, row=row, col=col)

        # add rectangle for candlestick pattern
        rect_start = subset.index[back_candles - num_candles + 1]    # First pattern candle
        rect_end = subset.index[back_candles]                           # Pattern candle

        # calculate extended x-coordinates for the rectangle
        x_left = rect_start - pd.Timedelta(hours=12)
        x_right = rect_end + pd.Timedelta(hours=12)
        
        rect_low = min(subset['Low'].iloc[back_candles-num_candles+1:back_candles+1]) * 0.998    # Extend 0.2% below
        rect_high = max(subset['High'].iloc[back_candles-num_candles+1:back_candles+1]) * 1.002  # Extend 0.2% above
        
        # create the rectangle trace
        rect = go.Scatter(
            x=[x_left, x_left, x_right, x_right, x_left],
            y=[rect_low, rect_high, rect_high, rect_low, rect_low],
            mode='lines',
            line=dict(color='red', width=1),
            fill='none',
            showlegend=False,
            hoverinfo='skip'    # No hover info for the rectangle
        )
        
        fig.add_trace(rect, row=row, col=col)
        
        # Update axes in each subplot
        fig.update_xaxes(title_text=None, showgrid=True, 
                        row=row, col=col, rangeslider_visible=False)
        fig.update_yaxes(title_text=None, showgrid=True, row=row, col=col)

    fig.update_layout(
        #autosize = True,
        height=300*n_rows,  # Adjust height based on number of rows
        width=1100,         # Fixed width
        showlegend=False,
        hovermode='x unified', 
        #margin=dict(t=20, b=20, l=20, r=20),
        title = dict(text = f'Examples of {pattern_name} detected in {ticker} stock',
                                    font=dict(size=18, color='red'),
                                    x=0.5)
        )

    fig.show()
