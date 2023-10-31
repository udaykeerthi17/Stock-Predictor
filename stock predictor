import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()
symbol = 'AAPL'
start_date = '2022-01-01'
end_date = '2022-10-25'
data = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

seq_length = 60
X, y = [], []
for i in range(len(scaled_data)-seq_length):
    X.append(scaled_data[i:i+seq_length])
    y.append(scaled_data[i+seq_length])

X, y = np.array(X), np.array(y)

split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]

model = Sequential([
    LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(units=50, activation='relu'),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=10, batch_size=32)

predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.figure(figsize=(12,6))
plt.plot(data.index[seq_length+split_index:], data['Close'][seq_length+split_index:], label='Actual Price')
plt.plot(data.index[seq_length+split_index:], predicted_prices, label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{symbol} Stock Price Prediction')
plt.legend()
plt.show()
