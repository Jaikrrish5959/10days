import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from prophet import Prophet

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Step 1: Load data from Excel/CSV and clean it.
    - Parse dates
    - Remove negative QTY (returned goods)
    - Trim dates dynamically up to the maximum date in the file
    """
    # 1. Error Check: Verify the raw file exists at the path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # 2. Error Check: Read the file, supporting both Excel (.xlsx) and CSV formats
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload a valid .xlsx or .csv.")
    except Exception as e:
        raise ValueError("Could not read file. Please upload a valid .xlsx or .csv.") from e

    # 3. Error Check: Verify required columns exist in the file
    required_cols = ['SlNo', 'INVOICEDATE', 'ITEMID', 'QTY']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Please check your file has: {', '.join(required_cols)}")

    # 4. Error Check: Ensure there are at least 5 products for statistical modeling
    unique_items = df['ITEMID'].dropna().unique()
    if len(unique_items) < 5:
        raise ValueError(f"Need at least 5 products. Found only {len(unique_items)}.")

    # 5. Data Prep: Parse dates and drop rows where date parsing failed
    df['INVOICEDATE'] = pd.to_datetime(df['INVOICEDATE'], errors='coerce')
    df = df.dropna(subset=['INVOICEDATE'])

    # 6. Data Cleaning: Filter out negative quantities (returns/cancellations)
    df_cleaned = df[df['QTY'] >= 0]
    if df_cleaned.empty:
        raise ValueError("No positive sales found after cleaning. Check your data.")

    # 7. Data Cleaning: Trim the dataset (start from Oct 1, 2024; end dynamically at max date in file)
    max_date = df_cleaned['INVOICEDATE'].max()
    df_trimmed = df_cleaned[
        (df_cleaned['INVOICEDATE'] >= '2024-10-01') & 
        (df_cleaned['INVOICEDATE'] <= max_date)
    ]
    if df_trimmed.empty:
        raise ValueError("No data found in the required period (starting Oct 2024).")

    # 8. Error Check: Verify we have at least 90 days of history to capture seasonality
    days_span = (df_trimmed['INVOICEDATE'].max() - df_trimmed['INVOICEDATE'].min()).days + 1
    if days_span < 90:
        raise ValueError(f"Need at least 90 days of history to train. Your file has only {days_span} days.")

    return df_trimmed

def aggregate_and_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Aggregate & Format
    - Group by date, sum QTY (sum of all product quantities)
    - Reindex dynamically from min to max date to fill missing calendar days with 0
    - Rename columns to 'ds' and 'y' (Prophet formats)
    """
    # Group rows by invoice date and sum the quantities across all products
    df_daily = df.groupby('INVOICEDATE')['QTY'].sum().reset_index()
    
    # Generate a complete date sequence with no missing days
    start_date = df['INVOICEDATE'].min()
    end_date = df['INVOICEDATE'].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Reindex (fill missing dates with 0 sales)
    df_daily = df_daily.set_index('INVOICEDATE').reindex(all_dates, fill_value=0).reset_index()
    
    # Rename columns to Prophet's expected names
    df_daily.columns = ['ds', 'y']
    return df_daily

def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 3: Train/Test Split (Time-Series Chronological Split)
    - Training: Oct 1, 2024 to Feb 28, 2026
    - Testing: Mar 1, 2026 to May 31, 2026 (3 full months)
    """
    train_df = df[df['ds'] <= '2026-02-28']
    test_df = df[(df['ds'] >= '2026-03-01') & (df['ds'] <= '2026-05-31')]
    return train_df, test_df

def fit_prophet(df: pd.DataFrame) -> Prophet:
    """
    Step 4: Fit Prophet Model
    - Enable weekly and yearly cyclical patterns
    - Configure Multiplicative seasonality (seasonal changes scale with growth trend)
    - Load Indian national holiday markers to adjust curves on holiday dates
    """
    # Instantiate the Prophet model with custom settings
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='multiplicative'
    )
    # Add Indian national holidays automatically (e.g. Diwali, Republic Day)
    model.add_country_holidays(country_name='IN')
    # Train/fit the model on the formatted dataset
    model.fit(df)
    return model

def evaluate_model(model: Prophet, test_df: pd.DataFrame, plot_path: str = 'evaluation_plot.png') -> tuple[float, float, float]:
    """
    Step 5: Evaluate model on the test set
    - Predict over test dates (Mar - May 2026)
    - Calculate RMSE, MAPE (skipping zero sales), and WAPE (retail standard for sparse data)
    - Generate a line plot comparing actual vs. predicted values
    """
    # Generate predictions on test dates
    forecast = model.predict(test_df)
    
    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values
    
    # Calculate RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate MAPE (Mean Absolute Percentage Error) - ignoring actuals of 0 to avoid division by zero
    nonzero_mask = y_true != 0
    if np.sum(nonzero_mask) > 0:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = np.nan
        
    # Calculate WAPE (Weighted Absolute Percentage Error) - recommended for zero-heavy retail data
    sum_actuals = np.sum(y_true)
    if sum_actuals > 0:
        wape = np.sum(np.abs(y_true - y_pred)) / sum_actuals * 100
    else:
        wape = np.nan

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['ds'], y_true, label='Actual Sales (y)', color='#1f77b4', linewidth=2)
    plt.plot(forecast['ds'], y_pred, label='Forecasted Sales (yhat)', color='#ff7f0e', linewidth=2)
    plt.fill_between(
        forecast['ds'], 
        forecast['yhat_lower'], 
        forecast['yhat_upper'], 
        color='#ff7f0e', 
        alpha=0.15, 
        label='95% Uncertainty Interval'
    )
    
    # Add titles, labels, and clean layout styles
    plt.title('Prophet Model Evaluation (Mar 2026 - May 2026)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Daily Quantity Sold', fontsize=12)
    plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#e0e0e0')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return rmse, mape, wape

def refit_and_forecast(df: pd.DataFrame, forecast_days: int = 90) -> tuple[Prophet, pd.DataFrame]:
    """
    Step 6: Refit and Forecast
    - Fit on the full historical dataset (Oct 2024 to present)
    - Generate forecast for the next N days
    """
    # Refit model on the entire dataset
    model = fit_prophet(df)
    
    # Create future dates table
    future = model.make_future_dataframe(periods=forecast_days, freq='D')
    
    # Run predictions
    forecast_full = model.predict(future)
    
    # Filter predictions to future dates only (greater than the last historical date)
    last_date = df['ds'].max()
    forecast_90 = forecast_full[forecast_full['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    return model, forecast_90

def prepare_product_series(df: pd.DataFrame, item_id: str) -> pd.DataFrame:
    """
    Filter transactions for a single product code (ITEMID) and reindex daily.
    - Zero-fills any dates where no transactions occurred.
    """
    # Filter to specific product
    df_item = df[df['ITEMID'] == item_id]
    
    # Group quantities daily
    df_daily = df_item.groupby('INVOICEDATE')['QTY'].sum().reset_index()
    
    # Create complete date range matching the overall cleaned data limits
    start_date = df['INVOICEDATE'].min()
    end_date = df['INVOICEDATE'].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Fill missing days with 0
    df_daily = df_daily.set_index('INVOICEDATE').reindex(all_dates, fill_value=0).reset_index()
    df_daily.columns = ['ds', 'y']
    return df_daily

def forecast_single_product(df_clean: pd.DataFrame, item_id: str, forecast_days: int = 10) -> tuple[Prophet, pd.DataFrame]:
    """
    Trains a Prophet model for a single product and generates a future forecast.
    - Clips predicted sales values to 0 (since sales cannot be negative).
    - Rounds floats to the nearest integer.
    """
    # Prep daily sales series
    df_series = prepare_product_series(df_clean, item_id)
    
    # Fit Prophet
    model = fit_prophet(df_series)
    
    # Forecast future days
    future = model.make_future_dataframe(periods=forecast_days, freq='D')
    forecast = model.predict(future)
    
    # Filter forecast values to future dates only
    last_date = df_series['ds'].max()
    forecast_future = forecast[forecast['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    
    # Clip negative values and round to integer
    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        forecast_future[col] = forecast_future[col].clip(lower=0).round(0).astype(int)
        
    return model, forecast_future

def forecast_all_products_parallel(df_clean: pd.DataFrame, forecast_days: int = 10, progress_callback=None) -> dict[str, pd.DataFrame]:
    """
    Runs Prophet forecasts for all unique products in parallel.
    - Uses ThreadPoolExecutor to train models concurrently across CPU cores.
    - Calls progress_callback (if provided) to update the dashboard progress bar.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    unique_items = sorted(df_clean['ITEMID'].dropna().unique())
    results = {}
    
    total_items = len(unique_items)
    completed = 0
    
    # Automatically allocate workers based on system cores (capped at 8)
    max_workers = min(os.cpu_count() or 4, 8)
    
    # Worker function to run inside threads
    def worker(item_id):
        try:
            _, fc = forecast_single_product(df_clean, item_id, forecast_days)
            return item_id, fc
        except Exception as e:
            print(f"Error forecasting {item_id}: {e}")
            return item_id, None

    # Run parallel threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, item): item for item in unique_items}
        for future in as_completed(futures):
            item_id, fc = future.result()
            if fc is not None:
                results[item_id] = fc
            completed += 1
            if progress_callback:
                progress_callback(completed, total_items)
                
    return results

def format_forecast_table(forecast_dict: dict[str, pd.DataFrame], forecast_unit: str = "Days") -> pd.DataFrame:
    """
    Consolidates product forecasts into a wide format table.
    - Days: Columns are individual dates (YYYY-MM-DD).
    - Weeks: Columns are weekly sales aggregates (Week 1, Week 2, etc.) created by summing daily predictions.
    """
    rows = []
    for item_id, fc_df in forecast_dict.items():
        row = {'ITEMID': item_id}
        
        if forecast_unit == "Days":
            # Direct mapping of daily dates
            for _, r in fc_df.iterrows():
                date_str = r['ds'].strftime('%Y-%m-%d')
                row[date_str] = r['yhat']
        else: # Weeks
            # Group daily rows by 7-day increments and sum values
            fc_df_sorted = fc_df.sort_values(by='ds').reset_index(drop=True)
            for i in range(len(fc_df_sorted) // 7):
                week_num = i + 1
                week_sum = fc_df_sorted.loc[i*7 : (i+1)*7 - 1, 'yhat'].sum()
                row[f"Week {week_num}"] = week_sum
                
        rows.append(row)
        
    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        # Arrange columns: place ITEMID first, then sort remaining dates/weeks
        cols = [c for c in df_res.columns if c != 'ITEMID']
        
        if forecast_unit == "Days":
            cols = sorted(cols)
        else:
            cols = sorted(cols, key=lambda x: int(x.split(' ')[1]))
            
        df_res = df_res[['ITEMID'] + cols]
        
        # Calculate row averages and sort product list descending (strongest sellers first)
        avg_col_name = "Avg daily QTY" if forecast_unit == "Days" else "Avg weekly QTY"
        df_res[avg_col_name] = df_res[cols].mean(axis=1).round(1)
        df_res = df_res.sort_values(by=avg_col_name, ascending=False)
        
    return df_res
