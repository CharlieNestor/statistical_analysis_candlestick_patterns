import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from astropy.stats import knuth_bin_width
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

def calculate_nbins(data: any) -> int:
    """
    Calculate the number of bins for a histogram using Freedman-Diaconis rule 
    """
    array = np.array(data)
    q1 = np.percentile(array, 25)
    q3 = np.percentile(array, 75)
    iqr = q3 - q1
    bin_width = (2 * iqr) / (array.shape[0] ** (1 / 3))
    bin_count = int(np.ceil((array.max() - array.min()) / bin_width))
    return int(bin_count)


def plot_return_distributions(returns: dict[int, list[float]], num_cols: int = 3):
    """
    Plot frequency distributions of returns for each candle using Plotly
    :param returns: Dictionary of returns. Keys are the number of candles, values are lists of returns
    :param num_cols: Number of columns in the subplot grid
    """
    num_periods = len(returns)
    num_rows = (num_periods + num_cols - 1) // num_cols
    
    fig = make_subplots(rows=num_rows, cols=num_cols, 
                        subplot_titles=[f"Returns after {period} candles" for period in returns.keys()],
                        vertical_spacing=0.1)
    
    for idx, (period, ret) in enumerate(returns.items()):
        row = idx // num_cols + 1
        col = idx % num_cols + 1

        # calculate the number of bins for the histogram
        #num_bins = calculate_nbins(ret)
        #num_bins = np.histogram_bin_edges(np.array(ret), bins='auto')
        width, bin_edges = knuth_bin_width(np.array(ret), return_bins=True)
        
        # calculate histogram data
        hist, bin_edges = np.histogram(ret, bins=bin_edges, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # plot histogram
        fig.add_trace(go.Bar(x=bin_centers, y=hist, name='', showlegend=False,
                             hoverinfo='x'), 
                    row=row, col=col)
        
        
        # fit a normal distribution to the data
        mu, std = np.mean(ret), np.std(ret)
        x = np.linspace(min(ret), max(ret), 100)
        p = stats.norm.pdf(x, mu, std)
        
        '''
        # Fit a normal distribution centered at 0 with the same standard deviation as the data
        std = np.std(ret)
        x = np.linspace(min(ret), max(ret), 100)
        p = stats.norm.pdf(x, 0, std)  # Note: mean is set to 0 here
        '''
        
        # plot the normal distribution
        fig.add_trace(go.Scatter(x=x, y=p, mode='lines', name='Gaussian Distribution',
                                line=dict(color='red'), showlegend=False, 
                                hoverinfo='name',
                            ), row=row, col=col)
        
        # update axes labels
        fig.update_xaxes(title_text=None, row=row, col=col)
        fig.update_yaxes(title_text=None, row=row, col=col)
        # center the x-axis range
        fig.update_xaxes(range=[np.percentile(ret, 2), np.percentile(ret, 99)], row=row, col=col)


    fig.update_layout(height=300*num_rows, width=350*num_cols, 
                    title_text=f"Return Distributions by Candle vs Gaussian Distribution. (Sample Size: {len(ret)})")
    fig.show()


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


def qq_plot(true_returns: dict[int, list[float]], generated_returns: dict[int, list[float]], num_cols: int = 3):
    """
    Create Q-Q plots comparing true pattern distribution vs generated distribution
    
    :param true_returns: Dictionary of true returns. Keys are days, values are lists of returns
    :param generated_returns: Dictionary of generated returns. Keys are days, values are lists of returns
    :param num_cols: Number of columns in the subplot grid
    """
    num_days = len(true_returns)
    num_rows = (num_days + num_cols - 1) // num_cols
    
    fig = make_subplots(rows=num_rows, cols=num_cols,
                        subplot_titles=[f"Q-Q Plot for Day {day}" for day in true_returns.keys()],
                        vertical_spacing=0.1)
    
    for idx, day in enumerate(true_returns.keys()):
        row = idx // num_cols + 1
        col = idx % num_cols + 1
        
        true_data = np.sort(true_returns[day])
        generated_data = np.sort(generated_returns[day])
        
        # Calculate quantiles
        n_true = len(true_data)
        n_generated = len(generated_data)
        
        quantiles_true = np.arange(1, n_true + 1) / (n_true + 1)
        quantiles_generated = np.arange(1, n_generated + 1) / (n_generated + 1)
        
        # Interpolate generated data to match true data quantiles
        generated_interpolated = np.interp(quantiles_true, quantiles_generated, generated_data)
        
        # Add scatter plot
        fig.add_trace(go.Scatter(x=true_data, y=generated_interpolated, mode='markers',
                                 name='', showlegend=False),
                      row=row, col=col)
        
        # Add diagonal line
        min_val = min(min(true_data), min(generated_interpolated))
        max_val = max(max(true_data), max(generated_interpolated))
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                                 name='y=x', line=dict(color='red'), showlegend=False),
                      row=row, col=col)
        
        # Update axes labels
        fig.update_xaxes(title_text="True Quantiles", row=row, col=col)
        fig.update_yaxes(title_text="Generated Quantiles", row=row, col=col)
    
    fig.update_layout(height=300*num_rows, width=350*num_cols,
                      title_text="Q-Q Plots: True vs Generated Returns")
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
                      title_text=f"{metric_name} Distributions by Day vs Normal Distribution. (Sample Size: {len(values)})")
    fig.show()