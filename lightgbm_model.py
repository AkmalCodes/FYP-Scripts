import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load and preprocess the dataset
df = pd.read_csv('Reprocessed_SalesKaggle_Full.csv', parse_dates=['Date'])

# Group by SKU and month
df = df.groupby(['SKU_number', pd.Grouper(key='Date', freq='M')])['SoldCount'].sum().reset_index()
df = df.sort_values(['SKU_number', 'Date'])

# Create lag features
for lag in [1, 2, 3]:
    df[f'lag_{lag}'] = df.groupby('SKU_number')['SoldCount'].shift(lag)

# Drop rows with missing lag values
df.dropna(inplace=True)

# Convert SKU_number to category
df['SKU_number'] = df['SKU_number'].astype('category')

# Define features
features = ['SKU_number', 'lag_1', 'lag_2', 'lag_3']
target = 'SoldCount'

results = []
skipped_skus = []

# Evaluate per SKU
for sku in df['SKU_number'].cat.categories:
    sku_df = df[df['SKU_number'] == sku].copy()

    # Define cutoff date
    cutoff_date = sku_df['Date'].max() - pd.DateOffset(months=6)
    train_df = sku_df[sku_df['Date'] <= cutoff_date]
    test_df = sku_df[sku_df['Date'] > cutoff_date]

    # Skip if too little training data or missing values
    if len(train_df) < 10 or train_df[features].isnull().any().any():
        skipped_skus.append(sku)
        continue

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    try:
        model = lgb.LGBMRegressor(
            random_state=42,
            min_data_in_leaf=5,
            min_data_in_bin=1,
            num_leaves=10
        )
        model.fit(X_train, y_train)
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
        print(f"⚠️ Error processing SKU {sku}: {e}")
        skipped_skus.append(sku)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('lightgbm_performance_summary.csv', index=False)

print("\n✅ LightGBM evaluation completed.")
print(f"🧪 Processed SKUs: {len(results)}")
print(f"⏭️ Skipped SKUs: {len(skipped_skus)}")
