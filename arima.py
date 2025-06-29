
import pandas as pd
import numpy as np
from pmdarima.arima import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('inventory_forecast_dataset.csv')
df['created_at'] = pd.to_datetime(df['created_at'])

# Calculate average monthly soldCount to use as scaling factor
monthly_avg = df.groupby(['product_id', pd.Grouper(key='created_at', freq='M')])['soldCount'].sum().mean()
print(f"📊 Scaling soldCount by average monthly sold count: {monthly_avg:.2f}")
df['soldCount'] = df['soldCount'] / monthly_avg

# Aggregate monthly sales per SKU
monthly_df = df.groupby(['product_id', pd.Grouper(key='created_at', freq='MS')])['soldCount'].sum().reset_index()

# Filter SKUs with enough data
sku_groups = monthly_df.groupby('product_id')
sku_metrics = []

for sku, sku_data in sku_groups:
    sku_data = sku_data.sort_values('created_at')
    sku_data = sku_data.set_index('created_at')

    if len(sku_data) < 24:
        continue

    train = sku_data['soldCount'].iloc[:-6]
    test = sku_data['soldCount'].iloc[-6:]

    try:
        # Use fixed ARIMA(1,1,1) model
        model = ARIMA(order=(1, 1, 1))
        model.fit(train)

        forecast = model.predict(n_periods=6)
        forecast = pd.Series(np.maximum(forecast, 0), index=test.index)

        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mape = np.mean(np.abs((test.values.flatten() - forecast.values) / test.values.flatten())) * 100

        print(f"\nSKU: {sku}")
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        print("Model order: (1, 1, 1)")

        sku_metrics.append({
            'SKU_number': sku,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Order': (1, 1, 1)
        })

        if sku in [sku_groups.ngroup().unique()[0]]:
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
metrics_df.to_csv('arima_fixed_order_performance.csv', index=False)

# Print overall metrics for the report
print("\n=== Overall Evaluation Summary ===")
print(f"Average MAE: {metrics_df['MAE'].mean():.2f}")
print(f"Average RMSE: {metrics_df['RMSE'].mean():.2f}")
print(f"Average MAPE: {metrics_df['MAPE'].mean():.2f}%")
print("Saved performance metrics to 'arima_fixed_order_performance.csv'")
