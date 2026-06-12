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
    - Remove negative QTY (returns)
    - Trim to Oct 2024 - May 2026
    """
    # 1. File parse error checking
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload a valid .xlsx or .csv.")
    except Exception as e:
        raise ValueError("Could not read file. Please upload a valid .xlsx or .csv.") from e

    # 2. Missing required columns checking
    required_cols = ['SlNo', 'INVOICEDATE', 'ITEMID', 'QTY']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Please check your file has: SlNo, INVOICEDATE, ITEMID, QTY")

    # 3. Fewer than 5 products checking
    unique_items = df['ITEMID'].dropna().unique()
    if len(unique_items) < 5:
        raise ValueError(f"Need at least 5 products. Found only {len(unique_items)}.")

    # 4. Parse INVOICEDATE as datetime
    df['INVOICEDATE'] = pd.to_datetime(df['INVOICEDATE'], errors='coerce')
    df = df.dropna(subset=['INVOICEDATE'])

    # 5. Remove negative QTY
    df_cleaned = df[df['QTY'] >= 0]
    if df_cleaned.empty:
        raise ValueError("No positive sales found after cleaning. Check your data.")

    # 6. Trim Oct 2024 to the maximum date in the file (dynamic baseline)
    max_date = df_cleaned['INVOICEDATE'].max()
    df_trimmed = df_cleaned[
        (df_cleaned['INVOICEDATE'] >= '2024-10-01') & 
        (df_cleaned['INVOICEDATE'] <= max_date)
    ]
    if df_trimmed.empty:
        raise ValueError("No data found in the required period (Oct 2024 - May 2026).")

    # 7. Fewer than 90 days of data checking
    days_span = (df_trimmed['INVOICEDATE'].max() - df_trimmed['INVOICEDATE'].min()).days + 1
    if days_span < 90:
        raise ValueError(f"Need at least 90 days of history to train. Your file has only {days_span} days.")

    return df_trimmed

def aggregate_and_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Aggregate & Format
    - Group by date, sum QTY
    - Reindex to fill missing days with 0
    - Rename columns to ds and y
    """
    # Group by date, sum QTY
    df_daily = df.groupby('INVOICEDATE')['QTY'].sum().reset_index()
    
    # Reindex to fill missing days dynamically from the min date to max date in the data
    start_date = df['INVOICEDATE'].min()
    end_date = df['INVOICEDATE'].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    df_daily = df_daily.set_index('INVOICEDATE').reindex(all_dates, fill_value=0).reset_index()
    
    # Rename columns to ds and y
    df_daily.columns = ['ds', 'y']
    return df_daily

def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 3: Train/Test Split
    - Training: Oct 1 2024 to Feb 28 2026
    - Test: Mar 1 2026 to May 31 2026
    """
    train_df = df[df['ds'] <= '2026-02-28']
    test_df = df[(df['ds'] >= '2026-03-01') & (df['ds'] <= '2026-05-31')]
    return train_df, test_df

def fit_prophet(df: pd.DataFrame) -> Prophet:
    """
    Step 4: Fit Prophet Model with specified settings
    - weekly_seasonality = True
    - yearly_seasonality = True
    - seasonality_mode = 'multiplicative'
    - country holidays = 'IN' (India)
    """
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='multiplicative'
    )
    # Add Indian public holidays
    model.add_country_holidays(country_name='IN')
    model.fit(df)
    return model

def evaluate_model(model: Prophet, test_df: pd.DataFrame, plot_path: str = 'evaluation_plot.png') -> tuple[float, float, float]:
    """
    Step 5: Evaluate model on test set
    - Predict over test dates (Mar - May 2026)
    - Compute RMSE, MAPE (ignoring zero actuals), and WAPE
    - Plot actual vs predicted to visually inspect fit
    """
    forecast = model.predict(test_df)
    
    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values
    
    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Compute MAPE (ignoring actuals that are 0 to avoid division by zero)
    nonzero_mask = y_true != 0
    if np.sum(nonzero_mask) > 0:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = np.nan
        
    # Compute WAPE (Weighted Absolute Percentage Error, useful when there are zeros)
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
        label='Uncertainty Interval (95%)'
    )
    
    # Add labels and formatting
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
    - Fit on full data (Oct 2024 - May 2026)
    - Generate forecast for next N days (Jun - Aug 2026)
    """
    # Refit on full dataset
    model = fit_prophet(df)
    
    # Make future dataframe for N days
    future = model.make_future_dataframe(periods=forecast_days, freq='D')
    
    # Predict
    forecast_full = model.predict(future)
    
    # Filter only future dates
    last_date = df['ds'].max()
    forecast_90 = forecast_full[forecast_full['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    return model, forecast_90

def prepare_product_series(df: pd.DataFrame, item_id: str) -> pd.DataFrame:
    """
    Filter data for a specific ITEMID and aggregate/reindex it to daily frequency.
    """
    df_item = df[df['ITEMID'] == item_id]
    df_daily = df_item.groupby('INVOICEDATE')['QTY'].sum().reset_index()
    
    # Reindex to fill missing days dynamically from the min date to max date in the data
    start_date = df['INVOICEDATE'].min()
    end_date = df['INVOICEDATE'].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    df_daily = df_daily.set_index('INVOICEDATE').reindex(all_dates, fill_value=0).reset_index()
    
    df_daily.columns = ['ds', 'y']
    return df_daily

def forecast_single_product(df_clean: pd.DataFrame, item_id: str, forecast_days: int = 10) -> tuple[Prophet, pd.DataFrame]:
    """
    Fits a Prophet model for a single product and generates a 10-day forecast.
    """
    df_series = prepare_product_series(df_clean, item_id)
    model = fit_prophet(df_series)
    
    future = model.make_future_dataframe(periods=forecast_days, freq='D')
    forecast = model.predict(future)
    
    # Extract only forecasted days
    last_date = df_series['ds'].max()
    forecast_future = forecast[forecast['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    
    # Clip forecasted values to 0 (sales cannot be negative) and round to integers
    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        forecast_future[col] = forecast_future[col].clip(lower=0).round(0).astype(int)
        
    return model, forecast_future

def forecast_all_products_parallel(df_clean: pd.DataFrame, forecast_days: int = 10, progress_callback=None) -> dict[str, pd.DataFrame]:
    """
    Runs Prophet forecasts for all unique products in parallel using ThreadPoolExecutor.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    unique_items = sorted(df_clean['ITEMID'].dropna().unique())
    results = {}
    
    total_items = len(unique_items)
    completed = 0
    
    # Limit worker count to prevent CPU/memory exhaustion
    max_workers = min(os.cpu_count() or 4, 8)
    
    def worker(item_id):
        try:
            _, fc = forecast_single_product(df_clean, item_id, forecast_days)
            return item_id, fc
        except Exception as e:
            print(f"Error forecasting {item_id}: {e}")
            return item_id, None

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

def format_forecast_table(forecast_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Consolidates product forecasts into a wide format table (rows: products, columns: dates).
    """
    rows = []
    for item_id, fc_df in forecast_dict.items():
        row = {'ITEMID': item_id}
        for _, r in fc_df.iterrows():
            date_str = r['ds'].strftime('%Y-%m-%d')
            row[date_str] = r['yhat']
        rows.append(row)
        
    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        # Reorder columns to place ITEMID first, and sort dates chronologically
        date_cols = sorted([c for c in df_res.columns if c != 'ITEMID'])
        df_res = df_res[['ITEMID'] + date_cols]
        # Sort rows by average forecasted quantity sold (descending)
        df_res['Avg daily QTY'] = df_res[date_cols].mean(axis=1).round(1)
        df_res = df_res.sort_values(by='Avg daily QTY', ascending=False)
        
    return df_res

