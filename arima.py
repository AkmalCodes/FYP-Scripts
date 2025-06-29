import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('inventory_forecast_dataset.csv')
df['created_at'] = pd.to_datetime(df['created_at'])

# Aggregate monthly sales per SKU
monthly_df = df.groupby(['product_id', pd.Grouper(key='created_at', freq='ME')])['soldCount'].sum().reset_index()

# Filter SKUs with enough data
sku_groups = monthly_df.groupby('product_id')
sku_metrics = []

for sku, sku_data in sku_groups:
    sku_data = sku_data.set_index('Date').sort_index()

    # Require at least 24 data points for decent train/test split
    if len(sku_data) < 24:
        continue

    train = sku_data['soldCount'].iloc[:-6]
    test = sku_data['soldCount'].iloc[-6:]

    try:
        # Fit ARIMA
        model = auto_arima(train, seasonal=False, stepwise=True, max_p=3, max_q=3, max_order=5, suppress_warnings=True)
        forecast = model.predict(n_periods=6)
        forecast = pd.Series(forecast, index=test.index)

        # Evaluate
        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mape = np.mean(np.abs((test.values.flatten() - forecast.values) / test.values.flatten())) * 100

        print(f"\nSKU: {sku}")
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        print("Model order:", model.order)

        sku_metrics.append({
            'product_id': sku,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Order': model.order
        })

        # (Optional) Plot results for one or two SKUs
        if sku in [sku_groups.ngroup().unique()[0]]:  # only first SKU
            plt.figure(figsize=(10, 4))
            plt.plot(train.index, train, label='Train')
            plt.plot(test.index, test, label='Test')
            plt.plot(forecast.index, forecast, label='Forecast', linestyle='--')
            plt.title(f'SKU {sku} Forecast vs Actual')
            plt.legend()
            plt.show()

    except Exception as e:
        print(f"Error for SKU {sku}: {e}")

# Save performance summary
metrics_df = pd.DataFrame(sku_metrics)
metrics_df.to_csv('arima_performance_summary.csv', index=False)
print("\nSaved performance metrics to 'arima_performance_summary.csv'")
