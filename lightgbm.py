import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv('Reprocessed_SalesKaggle_Full.csv', parse_dates=['Date'])

# Aggregate monthly sales per SKU
df = df.groupby(['SKU_number', pd.Grouper(key='Date', freq='M')])['SoldCount'].sum().reset_index()
df = df.sort_values(['SKU_number', 'Date'])

# Create lag features (lag_1, lag_2, lag_3)
for lag in [1, 2, 3]:
    df[f'lag_{lag}'] = df.groupby('SKU_number')['SoldCount'].shift(lag)

# Drop rows with missing lag values
df.dropna(inplace=True)

# Convert SKU_number to category
df['SKU_number'] = df['SKU_number'].astype('category')

# Define features and target
features = ['SKU_number', 'lag_1', 'lag_2', 'lag_3']
target = 'SoldCount'

# Evaluation results holder
results = []

# Loop per SKU to evaluate LightGBM accuracy per product
for sku in df['SKU_number'].cat.categories:
    sku_df = df[df['SKU_number'] == sku].copy()
    
    # Skip SKUs with insufficient rows
    if len(sku_df) < 12:
        continue
    
    cutoff_date = sku_df['Date'].max() - pd.DateOffset(months=6)
    train_df = sku_df[sku_df['Date'] <= cutoff_date]
    test_df = sku_df[sku_df['Date'] > cutoff_date]

    if len(test_df) == 0 or len(train_df) < 6:
        continue

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    try:
        # Train model
        model = lgb.LGBMRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        results.append({
            'SKU_number': sku,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        })

    except Exception as e:
        print(f"Error processing SKU {sku}: {e}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('lightgbm_performance_summary.csv', index=False)
print("\n✅ Saved LightGBM results to 'lightgbm_performance_summary.csv'")
