import pandas as pd
import plotly.graph_objects as go


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
        yaxis_title = 'Close Price',
        hovermode = 'x',
        autosize = True,
    )

    fig.update_yaxes(type='log')

    fig.show()

