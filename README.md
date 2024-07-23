# Statistical Approach to Candlestick Patterns

## Overview

This repository contains code and analysis for detecting and visualizing various candlestick patterns in stock market data. It's part of a larger project exploring the effectiveness of candlestick patterns in technical analysis using a data-driven approach.

For a detailed walkthrough of the analysis and methodology, check out our Medium articles:

1. [Candlestick Patterns in Trading: A Data-Driven Journey (Part 1)](https://medium.com/@carlo.baroni.89/candlestick-patterns-in-trading-a-data-driven-journey-c93ba8caae48)
2. [Candlestick Patterns in Trading: A Data-Driven Journey (Part 2)](https://medium.com/@carlo.baroni.89/candlestick-patterns-in-trading-a-data-driven-journey-6eee85aa1355)

These articles provide in-depth explanations and insights into the work done in this project.

## Why This Project?

Candlestick patterns are widely used in technical analysis, but their effectiveness is often debated. This project aims to:

1. Provide a robust method for programmatically detecting any candlestick pattern
2. Visualize these patterns across long-term stock data
3. Statistical analyze the results of these patterns 
4. Offer a comprehensive toolkit for traders and researchers to evaluate candlestick patterns


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


## Interactive Analysis Tool

This project now includes a Streamlit app (`pattern_streamlit.py`) that allows users to:

Select stocks from a predefined list or enter custom tickers
Choose from various candlestick patterns
Visualize pattern occurrences on stock charts
Run simulations to assess pattern effectiveness
View comparative performance metrics and significance heatmaps


## Future Work

This repository is part of an ongoing project. Future updates will include:

- Expansion of the pattern library
- Integration of machine learning techniques for pattern recognition
- Development of a backtesting framework for trading strategies based on candlestick patterns
- Optimization of the Streamlit app for better performance and user experience

I'm committed to evolving this project into a comprehensive, easy-to-use solution for candlestick pattern analysis in stock trading.

