# LSTM Forecasting Model Setup

# Install and import libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load Data
df = pd.read_csv('Reprocessed_SalesKaggle_Full.csv', parse_dates=['Date'])

# Aggregate monthly sales per SKU
monthly_df = df.groupby(['SKU_number', pd.Grouper(key='Date', freq='ME')])['SoldCount'].sum().reset_index()

# Define lag features
def create_lags(data, lags=3):
    for lag in range(1, lags + 1):
        data[f'lag_{lag}'] = data['SoldCount'].shift(lag)
    return data.dropna()

results = []

# Evaluate LSTM per SKU
for sku in monthly_df['SKU_number'].unique():
    sku_df = monthly_df[monthly_df['SKU_number'] == sku].copy()
    sku_df = create_lags(sku_df)

    # Ensure sufficient data points
    if len(sku_df) < 15:
        continue

    # Train-Test Split (last 6 months test)
    train = sku_df.iloc[:-6]
    test = sku_df.iloc[-6:]

    # Scale Data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[['SoldCount', 'lag_1', 'lag_2', 'lag_3']])
    test_scaled = scaler.transform(test[['SoldCount', 'lag_1', 'lag_2', 'lag_3']])

    X_train, y_train = train_scaled[:, 1:], train_scaled[:, 0]
    X_test, y_test = test_scaled[:, 1:], test_scaled[:, 0]

    # Reshape for LSTM (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # LSTM Model Definition
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train Model
    model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=0)

    # Predict
    predictions = model.predict(X_test).flatten()

    # Reverse scaling
    predictions_original = scaler.inverse_transform(
        np.column_stack((predictions, X_test.reshape(X_test.shape[0], X_test.shape[2])))
    )[:, 0]

    y_test_original = scaler.inverse_transform(
        np.column_stack((y_test, X_test.reshape(X_test.shape[0], X_test.shape[2])))
    )[:, 0]

    # Evaluation Metrics
    mae = mean_absolute_error(y_test_original, predictions_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
    mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100

    results.append({
        'SKU_number': sku,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    })

    print(f"Evaluated SKU {sku}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

# Save LSTM Evaluation Results
results_df = pd.DataFrame(results)
results_df.to_csv('lstm_performance_summary.csv', index=False)
print("\n✅ LSTM evaluation complete. Results saved to 'lstm_performance_summary.csv'.")
