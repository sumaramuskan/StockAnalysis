# StockAnalysis


This project predicts stock prices using historical data and compares predicted values with actual values. It utilizes Long Short-Term Memory (LSTM) models, a type of recurrent neural network (RNN), for time series prediction. The stock data is fetched using `yfinance`, and the analysis is visualized with `Matplotlib`.

## Project Overview

The project focuses on predicting the stock price of **Tata Consultancy Services (TCS.BO)** from **January 1, 2010** to **May 4, 2024**. The dataset is split into training and testing sets, and the LSTM model is trained using the closing price data. The model then predicts future prices, and a comparison is made between the predicted and actual stock prices.

## Features

- **Data Collection**: Stock data is collected using `yfinance`.
- **Data Preprocessing**: Historical stock data is cleaned and normalized.
- **Visualization**: Visualizes stock trends with 100-day and 200-day moving averages.
- **Modeling**: LSTM model with multiple layers for robust time series prediction.
- **Prediction**: Predicts stock prices and plots the predicted vs actual prices for analysis.

## Libraries Used

- **NumPy**: For numerical computations.
- **Pandas**: To handle stock data efficiently.
- **Matplotlib**: To visualize stock price trends.
- **TensorFlow**: For building and training the LSTM model.
- **yfinance**: To retrieve historical stock data.
- **sklearn.preprocessing.MinMaxScaler**: For scaling the data before feeding it to the model.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sumaramuskan/StockAnalysis.git
   ```
2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib yfinance scikit-learn tensorflow
   ```

## Data Preparation

1. **Data Retrieval**: Stock data for TCS is fetched using `yfinance` from the start date (Jan 1, 2010) to the end date (May 4, 2024).
   ```python
   data = yf.download('TCS.BO', start='2010-01-01', end='2024-05-04')
   ```
2. **Moving Averages**: 100-day and 200-day moving averages are calculated to smooth the data and identify trends.
   ```python
   ma100 = data['Close'].rolling(100).mean()
   ma200 = data['Close'].rolling(200).mean()
   ```
3. **Train-Test Split**: The dataset is split into 70% training data and 30% testing data.
   ```python
   data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.7)])
   data_testing = pd.DataFrame(data['Close'][int(len(data)*0.7):int(len(data))])
   ```

## Model Building

A sequential LSTM model is built using `TensorFlow` with multiple layers of LSTM and Dropout to prevent overfitting:
```python
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))
```

The model is compiled using the Adam optimizer and trained for 50 epochs:
```python
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50)
```

## Prediction and Visualization

Once trained, the model is used to predict stock prices, and the predictions are plotted alongside actual stock prices:
```python
plt.figure(figsize=(12,6))
plt.plot(data_testing, color='blue', label='Actual Stock Price')
plt.plot(predicted_stock_price, color='red', label='Predicted Stock Price')
plt.title('TCS Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
```

## Results

The LSTM model's predictions are plotted along with the actual stock prices, allowing for a visual comparison and analysis of the model’s performance.

## Future Work

- Adding more features such as volume, open, high, and low prices to improve model accuracy.
- Experimenting with different neural network architectures and hyperparameters.
- Testing the model with other stocks or indices for more robust performance.

---
