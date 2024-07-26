# I think one last thing we should do with your current project though is use the ACF and PACF to run time series predictions on a ton of stocks 
# and different time lengths and measure its effectiveness, then we can report this all in the Notion and label the project finished
# ut divide it by the starting price of the prediction so that the difference is scaled properly

import pacf_acf_calculator as pa
import yfinance as yf
import pandas as pd
import math

def analyze_prediction(ticker, start_date, end_training_date, start_date_prediction, end_date, forecast_period, interval, stock_data):
    '''will analyze how well the prediction fares against the actual stock closing price, it assumes that you give it the correct 
    forecast end/actual end'''
    # calculate the model order first
    pacf_order, pacf = pa.create_pacf(ticker, start_date, end_training_date, interval)
    acf_order, acf = pa.create_acf(ticker, start_date, end_training_date, interval)
    order = (pacf_order, 1, acf_order)

    # do the prediction
    ts_training = pd.Series(stock_data['Adj Close'][ticker][start_date : end_training_date])

    ts_actual = pd.Series(stock_data['Adj Close'][ticker][start_date_prediction : end_date])
    ts_prediction = pa.custom_model(ts_training, order, forecast_period)

    # make sure the lengths are the same
    # print(len(ts_actual))
    # print(len(ts_prediction))
    print(type(ts_actual))

    initial_value = ts_training[-1]
    differences = [ts_prediction[i].values[5:] - ts_actual[i].values[5:] for i in range(len(ts_prediction))]
    
    # return differences



if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df['Symbol'].tolist()
    
    stock_data = yf.download(tickers, start='2023-10-01', end='2024-07-17', interval='1d')
    
    diff = analyze_prediction('GOOG', '2023-10-01', '2024-07-02', '2024-07-02', '2024-07-17', 10, '1d', stock_data)
    print(diff)