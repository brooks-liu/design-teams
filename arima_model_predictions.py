import warnings
warnings.filterwarnings("ignore")
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

# this is the custom function I made to run an ARIMA model cleanly
# for more info on ARIMA model, see https://www.investopedia.com/terms/a/autoregressive-integrated-moving-average-arima.asp
def custom_model(data, order, periods):

    # data is a numpy array of data like stock price, it usually has a time index
    # order is a tuple of 3 numbers that represent the ARIMA order (kinda like settings for the ARIMA model)
    # periods is the number of periods you want to forecast into the future, so like 10 days or something
    return ARIMA(data, order=order).fit().forecast(steps=periods)


def arima_stock_plot(tick, shift, model, period):


    if period == 0:
        mth_stock_data = yf.download(tick, start='2023-10-02', end='2023-11-30', interval='1d')
        stk_ts_model = pd.Series(mth_stock_data['Adj Close']['2023-10-02':'2023-11-14'])
        stk_ts = pd.Series(mth_stock_data['Adj Close'])
        periods = 10
    else:
        yr_stock_data = yf.download(tick, start='2023-01-01', end='2023-11-30', interval='1d')
        stk_ts_model = pd.Series(yr_stock_data['Adj Close']['2023-01-01':'2023-07-01'])
        stk_ts = pd.Series(yr_stock_data['Adj Close'])
        periods = 100

    # you need to change the index of the pandas data frame because it was running some error, the ARIMA model works when it knows the index of the time series data
    stk_ts_model.index = pd.DatetimeIndex(stk_ts_model.index).to_period('D')

    # a bunch of different orders that I tested to see which one was the best
    order = [(1, 1, 1), (4, 1, 2), (40, 0, 1), (20, 1, 2), (10, 2, 1), (20, 1, 5), (15, 2, 7)]

    # here I make a list for all the forecasts and then I add each forecast to it
    stk_forecasts = []
    for i in range(len(order)):
        stk_forecasts.append(pd.concat([stk_ts_model, custom_model(stk_ts_model, order[i], periods)]))

    # here I plot the forecast for the model that was selected
    if model != -1:
        plt.plot(stk_forecasts[model].values[shift:])
    
    else:
        for i in range (len(stk_forecasts)):

            plt.plot(stk_forecasts[i].values[shift:])
    
    plt.plot(stk_ts.values[shift:])

    # I also return the forecasts so I can analyze them later
    if period == 0:
        return [stk_ts, stk_forecasts[1]] # tested to be optimal forecast for period = 0
    else:
        return [stk_ts, stk_forecasts[2]] # tested to be optimal forecast for period = 1


# this is creating a data frame which is like an excel sheet with data but you don't see it
df = pd.DataFrame()

tickers = ['GOOGL', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'KO', 'PG', 'LMT', 'DOL']

# these yfinance functions download stock data (opening price, closing price, volume, etc.) from Yahoo Finance online, from the specific dates with an interval
yr_stock_data = yf.download(tickers, start='2023-01-01', end='2024-07-02', interval='1d')
mth_stock_data = yf.download(tickers, start='2023-10-01', interval='1d')


import pacf_acf_calculator as pa

for ticker in tickers:
    ts_model = pd.Series(mth_stock_data['Adj Close'][ticker]['2023-10-01' : '2024-05-23'])
    ts = pd.Series(mth_stock_data['Adj Close'][ticker]['2023-10-01' : '2024-07-02'])
    acf_order, acf_values = pa.create_acf(ticker, '2023-10-01', '2024-05-23', '1d')
    pacf_order, pacf_values = pa.create_pacf(ticker, '2023-10-01', '2024-05-23', '1d')

    ts_model.index = pd.DatetimeIndex(ts_model.index).to_period('D')
    order = [(pacf_order, 1, acf_order)]
    periods = 10

    forecasts = []
    for i in range(1):
        forecasts.append(pd.concat([ts_model, custom_model(ts_model, order[i], periods)]))
    for i in range(len(forecasts)):
        plt.plot(forecasts[i].values[5:])

    plt.plot(ts.values[5:])
    plt.title(ticker)
    plt.show()
