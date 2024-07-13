import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from astropy.stats import knuth_bin_width
from plotly.subplots import make_subplots
from typing import Dict, Tuple


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
                    linecolor = 'black', mirror = True,
                    tickangle = -45,
                    )

    fig.update_yaxes(type = "log",            # this adds a logarithmic scale on the y axis
                    showgrid = True,
                    gridcolor = 'blue', griddash = "longdash",
                    showline = True, linewidth = 2, 
                    linecolor = 'black', mirror = True,
                    )

    fig.show()


def plot_close_with_patterns(data: pd.DataFrame, ticker: str, mask: pd.Series, pattern_name: str) -> None:
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
            x = date - pd.Timedelta(hours=48),
            y = 0.05,
            yref = "paper",
            text = date.strftime('%Y-%m-%d'),
            showarrow = False,
            textangle = -90,
            font = dict(size=7, color='red'),
        )

    fig.update_layout(
        title = dict(text = f'{ticker} Close Price with {pattern_name} occurrences',
                            font=dict(size=18, color='red'),
                            x=0.5,
                            ),
        xaxis_title = 'Date',
        yaxis_title = 'Close Price (log)',
        hovermode = 'x',
        autosize = True,
    )

    fig.update_xaxes(#showline = True, 
                    linewidth = 1, 
                    linecolor = 'black', mirror = True,
                    tickangle = -45,
                    )

    fig.update_yaxes(type='log',
                    linewidth = 1, 
                    linecolor = 'black', mirror = True,
                    )

    fig.show()


def plot_patterns(data: pd.DataFrame, mask: pd.Series, num_candles: int, ticker: str, pattern_name: str, max_candles=20, back_candles=5, max_subplots=12) -> None:
    """
    create a subplot candlestick chart for each pattern detected up to max_subplots, randomly selected
    :param data: the stock data
    :param mask: a boolean mask with True where patterns occur
    :param num_candles: the number of candles in the pattern
    :param ticker: the stock ticker
    :param pattern_name: the name of the pattern
    :param max_candles: the maximum number of candles to show in each subplot
    :param back_candles: the number of candles to show before the pattern
    """
    # find the dates where patterns occur
    pattern_dates = data.index[mask]
    
    # select random dates from the pattern_dates up to k instances
    if len(pattern_dates) <= max_subplots:
        section_dates = pattern_dates
    else:
        section_dates = random.sample(list(pattern_dates), k=max_subplots)
    section_dates = sorted(section_dates)
    
    # determine the number of rows and columns for the subplots
    n_plots = len(section_dates)
    n_cols = min(3, n_plots)        # Max 3 columns
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # create subplot figure
    subplot_titles = [date.strftime('%B %d, %Y') for date in section_dates]
    fig = make_subplots(rows=n_rows, cols=n_cols, 
                        vertical_spacing=0.1, horizontal_spacing=0.05,
                        subplot_titles=subplot_titles,
                        )
    
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
        if col == 1:
            fig.update_yaxes(title_text='Prices', showgrid=True, row=row, col=col)
        else:
            fig.update_yaxes(title_text=None, showgrid=True, row=row, col=col)

    # Customize the layout of titles of the subplots
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=16)

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

'''
def plot_original_stats(avg_returns: dict[int, float], median_returns: dict[int, float], win_rate: dict[int, float]) -> None:
    """
    Plot the average cumulative returns, median cumulative returns, and win rate over the future periods
    :param avg_returns: dictionary with the average cumulative returns for each period. Keys are the periods, value is the average returns
    :param median_returns: dictionary with the median cumulative returns for each period. Keys are the periods, value is the median returns
    :param win_rate: dictionary with the win rate for each period. Keys are the periods, value is the average win rates
    """

    fig = go.Figure()

    # add average cumulative returns trace
    fig.add_trace(
        go.Scatter(
            x=list(avg_returns.keys()),
            y=list(avg_returns.values()),
            name="Average Returns",
            mode='lines+markers',
            yaxis="y1",
            hoverinfo='x+y+name',
            line=dict(color='blue')
        )
    )

    # add median cumulative returns trace
    fig.add_trace(
        go.Scatter(
            x=list(median_returns.keys()),
            y=list(median_returns.values()),
            name="Median Returns",
            mode='lines+markers',
            yaxis="y1",
            hoverinfo='x+y+name',
            line=dict(color='red')
        )
    )

    # add win rate trace
    fig.add_trace(
        go.Scatter(
            x=list(win_rate.keys()),
            y=list(win_rate.values()),
            name="Win Rate",
            mode='lines+markers',
            yaxis="y2",
            hoverinfo='x+y+name',
            line=dict(color='green')
        )
    )

    # update layout for dual y-axes
    fig.update_layout(
        title=f"Cumulative Returns and Win Rate Over the next Days",
        xaxis_title="Future Lag (Days)",
        yaxis1=dict(
            title="Cumulative Returns",
        ),
        yaxis2=dict(
            title="Win Rate (%)",
            overlaying="y",
            side="right",
            range=[min(win_rate.values()) * 0.8, max(win_rate.values()) * 1.1]
        ),
        legend=dict(
            #x=1.05,    by right side
            #y=1,
            x=0.01,
            y=0.99,
            bordercolor="Black",
            borderwidth=1
        ),
    )

    fig.show()
'''


def plot_compared_metrics(pattern_avg_returns: dict[int, float], pattern_median_returns: dict[int, float], pattern_win_rate: dict[int, float],
                        base_avg_returns: dict[int, float], base_median_returns: dict[int, float], base_win_rate: dict[int, float],
                        confidence_intervals: Dict[str, Dict[int, Tuple[float, float]]], show_interval:str = 'average_return') -> None:
    """
    Plot the average cumulative returns, median cumulative returns, and win rate over the future periods
    for both pattern and base case, including confidence intervals for the base case.
    
    :param pattern_avg_returns: dictionary with the pattern average cumulative returns for each period. Keys are the periods, value is the average returns
    :param pattern_median_returns: dictionary with the pattern median cumulative returns for each period. Keys are the periods, value is the median returns
    :param pattern_win_rate: dictionary with the pattern win rate for each period. Keys are the periods, value is the average win rates
    :param base_avg_returns: dictionary with the base average cumulative returns for each period. Keys are the periods, value is the average returns
    :param base_median_returns: dictionary with the base median cumulative returns for each period. Keys are the periods, value is the median returns
    :param base_win_rate: dictionary with the base win rate for each period. Keys are the periods, value is the average win rates
    :param confidence_intervals: dictionary with confidence intervals for each metric. Keys are the metric names, values are dictionaries with keys as periods and values as tuples of confidence intervals
    """

    fig = go.Figure()

    # add pattern average cumulative returns trace
    fig.add_trace(
        go.Scatter(
            x=list(pattern_avg_returns.keys()),
            y=list(pattern_avg_returns.values()),
            name="Pattern Average Returns",
            mode='lines+markers',
            yaxis="y1",
            hoverinfo='x+y+name',
            line=dict(color='blue')
        )
    )

    # add base average cumulative returns trace
    fig.add_trace(
        go.Scatter(
            x=list(base_avg_returns.keys()),
            y=list(base_avg_returns.values()),
            name="Base Average Returns",
            mode='lines+markers',
            yaxis="y1",
            hoverinfo='x+y+name',
            line=dict(color='blue', dash='dash')
        )
    )

    # add pattern median cumulative returns trace
    fig.add_trace(
        go.Scatter(
            x=list(pattern_median_returns.keys()),
            y=list(pattern_median_returns.values()),
            name="Pattern Median Returns",
            mode='lines+markers',
            yaxis="y1",
            hoverinfo='x+y+name',
            line=dict(color='red')
        )
    )

     # add base median cumulative returns trace
    fig.add_trace(
        go.Scatter(
            x=list(base_median_returns.keys()),
            y=list(base_median_returns.values()),
            name="Base Median Returns",
            mode='lines+markers',
            yaxis="y1",
            hoverinfo='x+y+name',
            line=dict(color='red', dash='dash')
        )
    )

    # add pattern win rate trace
    fig.add_trace(
        go.Scatter(
            x=list(pattern_win_rate.keys()),
            y=list(pattern_win_rate.values()),
            name="Pattern Win Rate",
            mode='lines+markers',
            yaxis="y2",
            hoverinfo='x+y+name',
            line=dict(color='green')
        )
    )

    # add base win rate trace
    fig.add_trace(
        go.Scatter(
            x=list(base_win_rate.keys()),
            y=list(base_win_rate.values()),
            name="Base Win Rate",
            mode='lines+markers',
            yaxis="y2",
            hoverinfo='x+y+name',
            line=dict(color='green', dash='dash')
        )
    )

    if show_interval == 'average_return':
        # Add confidence intervals for base metrics
        for day in base_avg_returns.keys():
            # Average Returns Confidence Interval
            avg_ci_lower, avg_ci_upper = confidence_intervals['average_return'][day]
            fig.add_trace(
                go.Scatter(
                    x=[day, day],
                    y=[avg_ci_lower, avg_ci_upper],
                    mode='lines+markers',
                    line=dict(color='blue', dash='dot', width=1),
                    marker=dict(size=4, symbol='line-ew-open', color='blue'),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
    elif show_interval == 'median_return':
        for day in base_median_returns.keys():
            # Median Returns Confidence Interval
            median_ci_lower, median_ci_upper = confidence_intervals['median_return'][day]
            fig.add_trace(
                go.Scatter(
                    x=[day, day],
                    y=[median_ci_lower, median_ci_upper],
                    mode='lines+markers',
                    line=dict(color='red', dash='dot', width=1),
                    marker=dict(size=4, symbol='line-ew-open', color='red'),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )

    elif show_interval == 'win_rate':
        for day in base_win_rate.keys():
            # Win Rate Confidence Interval
            win_rate_ci_lower, win_rate_ci_upper = confidence_intervals['win_rate'][day]
            fig.add_trace(
                go.Scatter(
                    x=[day, day],
                    y=[win_rate_ci_lower, win_rate_ci_upper],
                    mode='lines',
                    line=dict(color='green', dash='dot'),
                    yaxis="y2",
                    showlegend=False,
                    hoverinfo='skip'
                )
            )

    # update layout for dual y-axes
    fig.update_layout(
        title=f"Cumulative Returns and Win Rate Over the next Days",
        xaxis_title="Future Lag (Days)",
        yaxis1=dict(
            title="Cumulative Returns",
        ),
        yaxis2=dict(
            title="Win Rate (%)",
            overlaying="y",
            side="right",
            range=[min(min(base_win_rate.values()), min(pattern_win_rate.values())) * 0.8, 
                   max(max(base_win_rate.values()), max(pattern_win_rate.values())) * 1.1],
            showgrid=False
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bordercolor="Black",
            borderwidth=1
        ),
        #autosize=True,
        height=600,
        width=1100,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        #yaxis2=dict(showgrid=False)
    )

    fig.show()




def plot_metric_distributions(metrics: dict[str, dict[int, np.ndarray]], metric_name: str, num_cols: int = 3):
    """
    Plot frequency distributions of a specific metric for each day using Plotly
    
    :param metrics: Dictionary of metrics. Keys are metric names, values are dictionaries with days as keys and numpy arrays of values as values
    :param metric_name: Name of the metric to plot
    :param num_cols: Number of columns in the subplot grid
    """
    if isinstance(metric_name, str):
        metric_name = metric_name.lower()
    else:
        raise ValueError("metric_name should be a string.")
    metric_data = metrics[metric_name]
    num_days = len(metric_data)
    num_rows = (num_days + num_cols - 1) // num_cols
    
    fig = make_subplots(rows=num_rows, cols=num_cols,
                        subplot_titles=[f"{metric_name} after {day} days" for day in metric_data.keys()],
                        vertical_spacing=0.1)
    
    for idx, (day, values) in enumerate(metric_data.items()):
        row = idx // num_cols + 1
        col = idx % num_cols + 1
        
        # Calculate histogram data
        hist, bin_edges = np.histogram(values, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot histogram
        fig.add_trace(go.Bar(x=bin_centers, y=hist, name='', showlegend=False,
                             hoverinfo='x'),
                      row=row, col=col)
        
        # Fit a normal distribution to the data
        mu, std = np.mean(values), np.std(values)
        x = np.linspace(min(values), max(values), 100)
        p = stats.norm.pdf(x, mu, std)
        
        # Plot the normal distribution
        fig.add_trace(go.Scatter(x=x, y=p, mode='lines', name='Normal Distribution',
                                 line=dict(color='red'), showlegend=False,
                                 hoverinfo='name'),
                      row=row, col=col)
        
        # Update axes labels
        fig.update_xaxes(title_text=None, row=row, col=col)
        fig.update_yaxes(title_text=None, row=row, col=col)
        
        # Center the x-axis range
        fig.update_xaxes(range=[np.percentile(values, 2), np.percentile(values, 98)], row=row, col=col)
    
    fig.update_layout(height=300*num_rows, width=350*num_cols,
                      title_text=f"{metric_name} Distributions by Day vs Normal Distribution. (Sample size: {len(values)})")
    
    fig.show()



