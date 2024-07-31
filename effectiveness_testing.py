'''Some testing: 
From 2023-10-01 to 2024-07-02 as training data, 2024-07-02 to 2024-07-17 as the ten day prediction,  '''

import pacf_acf_calculator as pa
import yfinance as yf
import pandas as pd
import math
import statistics as st

# to ignore warnings i was getting:
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html


def analyze_prediction(ticker, start_date, end_training_date, start_date_prediction, end_date, forecast_period, interval, stock_data):
    '''will analyze how well the prediction fares against the actual stock closing price, it assumes that you give it the correct 
    forecast end/actual end, start_date_prediction and end_training_date are the same'''
    # calculate the model order first
    pacf_order, pacf = pa.create_pacf(ticker, start_date, end_training_date, interval)
    acf_order, acf = pa.create_acf(ticker, start_date, end_training_date, interval)
    order = (pacf_order, 1, acf_order)

    # do the prediction
    ts_training = pd.Series(stock_data['Adj Close'][ticker][start_date : end_training_date])
    
    ts_training.index = pd.DatetimeIndex(ts_training.index).to_period('D')

    ts_actual = pd.Series(stock_data['Adj Close'][ticker][start_date_prediction : end_date])
    ts_prediction = pa.custom_model(ts_training, order, forecast_period)

    # make sure the lengths are the same
    # print(len(ts_actual))
    # print(len(ts_prediction))
    # print(type(ts_actual))

    initial_value = ts_training.iloc[-1]
    differences = [ts_prediction.iloc[i] - ts_actual.iloc[i] for i in range(len(ts_prediction))]
    
    return [differences, initial_value, [value for value in ts_actual.values]]

def compare_to_no_prediction(differences, initial_value, actual_values):
    if len(differences) != len(actual_values):
        print("Please input values for the same time period")
        return None
    
    accuracy = []

    for i in range(len(differences)):
        accuracy.append(abs(actual_values[i] - initial_value) - abs(differences[i]))
    
    return accuracy/initial_value * 100


if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df['Symbol'].tolist()

    tickers.remove('BF.B')
    tickers.remove('BRK.B')
    
    stock_data = yf.download(tickers, start='2023-10-01', end='2024-07-17', interval='1d')
    
    diff = analyze_prediction('GOOG', '2023-10-01', '2024-07-02', '2024-07-02', '2024-07-17', 10, '1d', stock_data)
    # print(diff)
    # print(type(diff[2]))

    # accuracy = compare_to_no_prediction(diff[0], diff[1], diff[2])
    # print(accuracy)

    store_accuracy = []

    for ticker in tickers:
        diff = analyze_prediction(ticker, '2023-10-01', '2024-07-02', '2024-07-02', '2024-07-17', 10, '1d', stock_data)
        accuracy = compare_to_no_prediction(diff[0], diff[1], diff[2])
        store_accuracy.append((ticker, accuracy))
    
    mean_percent = st.mean([accuracy[1] for accuracy in store_accuracy])
    print(mean_percent)

    

