import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import yfinance as yf
import math
from statsmodels.tsa.arima.model import ARIMA

def create_pacf(ticker, startdate, enddate, interval):
    data = yf.download(ticker, startdate, enddate, interval=interval)
    close_price = pd.Series(data['Adj Close'])

    # pacf_values = close_price.values
    values = close_price.to_numpy()

    #print(values)

    size = close_price.size
    if size == 0:
        return 0, 0
    significance = 1.96/math.sqrt(size)

    pacf_values = sm.tsa.stattools.pacf(values, nlags=20, method='yw')
    order = 0
    for value in pacf_values:
        if value > significance:
            order += 1

    return order, pacf_values

def create_acf(ticker, startdate, enddate, interval):
    data = yf.download(ticker, startdate, enddate, interval=interval)
    close_price = pd.Series(data['Adj Close'])

    # acf_values = close_price.values
    values = close_price.to_numpy()

    #print(values)

    size = close_price.size

    if size == 0:
        return 0, 0
    significance = 1.96/math.sqrt(size)

    acf_values, confidence = sm.tsa.stattools.acf(values, nlags=20, alpha=0.05, fft=True)

    order = 0
    for value in acf_values:
        if value > significance:
            order += 1
    return order, acf_values

def custom_model(data, order, periods):

    # data is a numpy array of data like stock price, it usually has a time index
    # order is a tuple of 3 numbers that represent the ARIMA order (kinda like settings for the ARIMA model)
    # periods is the number of periods you want to forecast into the future, so like 10 days or something
    return ARIMA(data, order=order).fit().forecast(steps=periods)

def plot_close_predication(tick, start_date, end_date, model_order, forecast_period):
    stock_data = yf.download(tick, start=start_date, end=end_date, interval='1d')
    stock_ts_prediction = pd.Series(stock_data['Adj Close'])
    stock_data = yf.download(tick, start_date, end_date, interval='1d')

    stock_ts_prediction.index = pd.DatetimeIndex(stock_ts_prediction.index).to_period('D')
    stock_ts = pd.Series(stock_data['Adj Close'])

    forecast = [pd.concat([stock_ts_prediction, custom_model(stock_ts_prediction, model_order, forecast_period)])]
    plt.figure(figsize=(14, 8))
    plt.plot(forecast[0].values[5:], label='Predicted Close Price', color='orange')
    plt.plot(stock_ts.values[5:], label='Actual Close Price', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    return forecast[0]

if __name__ == "__main__":
    pacf_order, pacf_values = create_pacf('NVDA', '2024-04-01', '2024-06-20', '1d')
    acf_order, acf_values = create_acf('NVDA', '2024-04-01', '2024-06-20', '1d')
    print(pacf_order)
    print(acf_order)

    plt.figure(figsize=(10, 6))
    plt.stem(range(len(pacf_values)), pacf_values)
    plt.title('Partial Autocorrelation Function')
    plt.xlabel('Lag')
    plt.ylabel('PACF')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.stem(range(len(acf_values)), acf_values)
    plt.title('Autocorrelation Function')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.show()



