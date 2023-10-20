import pandas as pd


def process_options_data(options_data, stock_prices_data, cp_flag="C"):
    # Convert date columns to datetime
    options_data['date'] = pd.to_datetime(options_data['date'])
    stock_prices_data['Date'] = pd.to_datetime(stock_prices_data['Date'])

    # Merge options and stock prices data based on date
    merged_data = options_data.merge(stock_prices_data, left_on='date', right_on='Date', how="inner")

    # Calculate time to expiration in years
    merged_data['exdate'] = pd.to_datetime(merged_data['exdate'])
    merged_data['time_to_expiration'] = (merged_data['exdate'] - merged_data['date']).dt.total_seconds() / (
                24 * 60 * 60 * 365)

    columns_to_drop = ['Date', 'Open', 'High', 'Low', 'Volume', 'date', 'exdate', 'volume', 'delta', 'gamma', 'vega',
                       'theta', 'optionid', 'ticker', 'index_flag', 'issuer', 'exercise_style']
    merged_data.drop(columns=columns_to_drop, inplace=True)

    merged_data = merged_data.dropna()

    # Filter call options
    processed_options_data = merged_data[merged_data['cp_flag'] == cp_flag]
    processed_options_data.drop(columns='cp_flag', inplace=True)

    processed_options_data = processed_options_data.reset_index()

    return processed_options_data
