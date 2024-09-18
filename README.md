# Statistical Approach to Candlestick Patterns

## Overview

This repository offers a data-driven toolkit for analyzing candlestick patterns in stock market data. It provides robust methods for detecting patterns, visualizing them across long-term data, and statistically analysing their effectiveness.

For a detailed walkthrough of the analysis and methodology, check out our Medium articles:

1. [Candlestick Patterns in Trading: A Data-Driven Journey (Part 1)](https://medium.com/@carlo.baroni.89/candlestick-patterns-in-trading-a-data-driven-journey-c93ba8caae48)
2. [Candlestick Patterns in Trading: A Data-Driven Journey (Part 2)](https://medium.com/@carlo.baroni.89/candlestick-patterns-in-trading-a-data-driven-journey-6eee85aa1355)


## Features

- Programmatic detection of various candlestick patterns
- Interactive visualization of patterns on stock charts
- Statistical analysis of pattern returns
- Monte Carlo simulations for assessing pattern effectiveness
- Pattern effectiveness analysis


## Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/CharlieNestor/statistical_analysis_candlestick_patterns.git
   ```

2. Install the required packages in a new virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
    ```sh
    streamlit run pattern_streamlit.py
    ```


## Usage

The Streamlit app provides an interactive interface for analysing candlestick patterns. It is recommended to set the background color to white for better visibility.

1. Select a stock from the dropdown or enter a custom ticker.
2. Choose a candlestick pattern to analyze.
3. Use the checkboxes to display:
   - Stock chart with pattern occurrences
   - Examples of individual pattern instances
4. Click "Run Simulation" to perform statistical analysis.
5. View the results, including:
   - Pattern vs. Base Case Performance chart
   - Significance Heatmap


## File Structure

- `patterns.py`: Contains functions for programmatically define various candlestick patterns
- `loader.py`: Utilities for loading and preprocessing stock data
- `plots.py`: Functions for creating interactive visualizations
- `analysis.py`: Functions for statistical analysis of pattern returns
- `pattern_stock.py`: Defines the PatternStock class which will be used int the Streamlit app
- `pattern_streamlit.py`: Streamlit app for interactive pattern analysis
- `Pattern_Medium_01.ipynb`: Jupyter notebook showcasing the code journey related to the first Medium publication
- `Pattern_Medium_02.ipynb`: Jupyter notebook showcasing the code and analysis journey described in the second Medium publication
- `Pattern_playground.ipynb`: Jupyter notebook for those who wants to directly play with the patterns and the functionalities


## Future Work

This repository is part of an ongoing project. Future updates will include:

- Expansion of the pattern library
- Integration of machine learning techniques for pattern recognition
- Development of a backtesting framework for trading strategies based on candlestick patterns
- Optimization of the Streamlit app for better performance and user experience

I'm committed to evolving this project into a comprehensive, easy-to-use solution for candlestick pattern analysis in stock trading.

