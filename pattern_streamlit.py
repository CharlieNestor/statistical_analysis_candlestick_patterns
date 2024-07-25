import os
import time
import random
import threading
import plots as pl
import patterns as pt
import analysis as an
import numpy as np
import pandas as pd
import streamlit as st
import plotly.io as pio
from pattern_stock import PatternStock

# Avoid Streamlit template which would override the background color of the plotly chart
pio.templates.default = "none"


def load_tickers_from_file():
    """
    Load tickers from a file. The function first checks if 'tickers.txt' or 'tickers.csv' exists, 
    and if so, reads tickers from the file.
    """
    if os.path.exists('tickers.txt'):
        with open('tickers.txt', 'r') as f:
            return [line.strip() for line in f if line.strip()]
    elif os.path.exists('tickers.csv'):
        return pd.read_csv('tickers.csv', header=None)[0].tolist()
    return []

def sample_tickers(tickers, n=50):
    return random.sample(tickers, min(n, len(tickers)))

@st.cache_data
def load_stock_data(ticker):
    """
    Load stock data and cache the result.
    This function will only re-run when the ticker changes.
    """
    try:
        stock = PatternStock(ticker)
        stock.load_data()
        return stock, None
    except ValueError as e:
        return None, str(e)
    
@st.cache_data
def apply_pattern(stock, pattern_name):
    """
    Apply a pattern to the stock data and cache the result.
    This function will only re-run when the stock or pattern changes.
    """
    if pattern_name not in stock.pattern_data:
        stock.apply_pattern(pattern_name)
        stock.calculate_metrics(pattern_name)
    return stock.get_data_for_plotting(pattern_name)


def run_simulation_with_progress(stock, pattern_name):
    """
    Run the simulation with a progress animation, using threading to separate
    the actual computation from the progress display.
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    details_text = st.empty()
    
    n_iterations = 1000
    simulation_complete = threading.Event()
    
    def run_actual_simulation():
        stock.run_simulation(pattern_name, n_iterations=n_iterations)
        simulation_complete.set()
    
    # Start the actual simulation in a separate thread
    simulation_thread = threading.Thread(target=run_actual_simulation)
    simulation_thread.start()
    
    # Display progress animation
    for i in range(n_iterations):
        if simulation_complete.is_set():
            break
        
        progress = (i + 1) / n_iterations
        progress_bar.progress(progress)
        status_text.text(f"Simulation in progress: {int(progress * 100)}% complete")
        
        if i == 0:
            details_text.text("Initializing simulation...")
        elif i < n_iterations - 1:
            if i % 100 == 0:
                details_text.text(f"Processing iteration {i+1}...")
        else:
            details_text.text("Finalizing results...")
        
        time.sleep(0.01)  # Small delay to prevent UI freezing
    
    # Wait for the actual simulation to complete if it hasn't already
    simulation_thread.join()
    
    # Ensure the progress bar is full
    progress_bar.progress(1.0)
    status_text.text("Simulation completed!")
    details_text.text("Generating final reports...")
    time.sleep(1)  # Give a moment for the user to see the completion message
    details_text.empty()  # Clear the details text



# Set page config
st.set_page_config(page_title="Candlestick Pattern Analysis", layout="wide")

# Title
st.title("Candlestick Pattern Analysis")

# Sidebar for user input
st.sidebar.header("Settings")

# Initialize session state for tickers and load tickers from file
if 'all_tickers' not in st.session_state:
    file_tickers = load_tickers_from_file()
    if file_tickers:
        if len(file_tickers) > 50:
            selected_tickers = sample_tickers(file_tickers)
        else:
            selected_tickers = file_tickers
    else:
        selected_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "JPM", "JNJ", "V"]

    st.session_state.all_tickers = sorted(selected_tickers)

# Initialize session state for error tracking and current ticker
if 'error' not in st.session_state:
    st.session_state.error = None
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None

# Function to reset error state
def reset_error_state():
    st.session_state.error = None

# Initialize stock instances dictionary in session state
if 'stock_instances' not in st.session_state:
    st.session_state.stock_instances = {}

# Stock selection
ticker = st.sidebar.selectbox("Select a stock:", st.session_state.all_tickers, index=None)
custom_ticker = st.sidebar.text_input("Or enter a custom stock ticker:")

# Check if custom ticker is valid
if custom_ticker:
    if not custom_ticker.isalpha():
        st.sidebar.error("Invalid ticker. Please enter only alphabetic characters.")
    else:
        ticker = custom_ticker.upper()

# Pattern selection
pattern_names = list(pt.patterns.keys())
pattern_name = st.sidebar.selectbox("Select a candlestick pattern:", pattern_names)

# Load data only when ticker changes
if ticker != st.session_state.current_ticker:
    reset_error_state()  # Reset error state when trying a new ticker
    stock, error = load_stock_data(ticker)
    if error:
        st.error(f"Error: {error}")
        st.stop()
    else:
        st.session_state.current_ticker = ticker
        st.session_state.current_stock = stock

# Only proceed if we have valid data
if ticker and pattern_name and 'current_stock' in st.session_state:

    current_stock = st.session_state.current_stock

    # Display basic stock information
    stock_info = current_stock.info
    st.subheader(f"Selected Stock: {stock_info.get('longName', ticker)}")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
    with col2:
        st.write(f"**Market Cap:** {stock_info.get('marketCap', 'N/A')}")
        st.write(f"**Market Beta:** {stock_info.get('beta', 'N/A')}")

    # Apply pattern if not already applied
    if pattern_name not in current_stock.pattern_data:
        current_stock.apply_pattern(pattern_name)
        current_stock.calculate_metrics(pattern_name)

    # Display number of pattern instances with color-coded background
    num_instances = current_stock.pattern_data[pattern_name]['dim_pattern']
    if num_instances > 60:
        st.success(f"Number of pattern instances: {num_instances}")
    elif num_instances > 30:
        st.warning(f"Number of pattern instances: {num_instances}")
    else:
        st.error(f"Number of pattern instances: {num_instances}")

    # Checkboxes to show/hide pattern charts
    show_close_chart = st.sidebar.checkbox("Show Pattern with Close Prices", value=False)
    show_pattern_examples = st.sidebar.checkbox("Show Examples of Patterns", value=False)

    # Display charts based on checkbox states
    plot_data = current_stock.get_data_for_plotting(pattern_name)
    
    if plot_data:
        if show_close_chart:
            # Display the stock chart with patterns
            st.header("Stock Chart with Pattern Occurrences")
            fig = pl.plot_close_with_patterns(plot_data['df'], ticker, plot_data['mask'], 
                                              plot_data['pattern_name'], streamlit=True)
            fig.update_layout(plot_bgcolor="#f0f2f6")
            st.plotly_chart(fig, use_container_width=True, theme=None)

        if show_pattern_examples:
            # Display individual pattern occurrences
            st.header("Individual Pattern Occurrences")
            fig = pl.plot_patterns(plot_data['df'], plot_data['mask'], plot_data['pattern_info']['candles'], 
                                   ticker, plot_data['pattern_name'], streamlit=True)
            fig.update_layout(plot_bgcolor="#f0f2f6")
            st.plotly_chart(fig, use_container_width=True, theme=None)

    # Button to run simulation
    if st.sidebar.button("Run Simulation"):
        run_simulation_with_progress(current_stock, pattern_name)
        st.success(f"Simulation completed for {ticker} - {pattern_name}")

    # Display simulation results if available
    plot_data = current_stock.get_data_for_plotting(pattern_name)
    
    if plot_data and 'confidence_intervals' in plot_data:
        # Plot compared metrics
        st.header("Pattern vs Base Case Performance")
        fig = pl.plot_compared_metrics(plot_data['pattern_metrics'], plot_data['confidence_intervals'], 
                                       plot_data['pattern_name'], streamlit=True)
        st.plotly_chart(fig, use_container_width=True)

        # Display significance heatmap
        st.header("Significance Heatmap")
        fig = pl.plot_significance_heatmap(plot_data['significance_table'], plot_data['pattern_name'], 
                                           streamlit=True)
        st.pyplot(fig)

else:
    st.info("Select a stock and a pattern to begin analysis.")


st.sidebar.info("Select a stock and a pattern, then use the checkbox to show/hide pattern charts. Click 'Run Simulation' for detailed statistical analysis.")
