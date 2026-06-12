import os
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pipeline import (
    load_and_clean_data,
    aggregate_and_format,
    split_train_test,
    fit_prophet,
    evaluate_model,
    refit_and_forecast
)

def main():
    parser = argparse.ArgumentParser(description="Prophet Sales Forecasting Pipeline")
    parser.add_argument(
        "--file", 
        type=str, 
        default=r"C:\Users\Jaikrish\Desktop\10days\SAMPLE SALES DATA.xlsx",
        help="Path to the raw sales data Excel or CSV file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output",
        help="Directory to save forecast output files and plots"
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("               PROPHET SALES FORECASTING PIPELINE")
    print("=" * 60)
    print(f"Input file: {args.file}")
    print(f"Output directory: {args.output_dir}\n")

    # Step 1: Load & Clean
    print("[Step 1] Loading and cleaning data...")
    try:
        df_cleaned = load_and_clean_data(args.file)
        print(f"  Successfully loaded raw data.")
        print(f"  Cleaned rows: {len(df_cleaned)}")
        print(f"  Cleaned date range: {df_cleaned['INVOICEDATE'].min().strftime('%Y-%m-%d')} to {df_cleaned['INVOICEDATE'].max().strftime('%Y-%m-%d')}")
        print(f"  Unique products: {df_cleaned['ITEMID'].nunique()}")
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Aggregate & Format
    print("\n[Step 2] Aggregating sales daily and formatting for Prophet...")
    df_daily = aggregate_and_format(df_cleaned)
    print(f"  Formatted data shape: {df_daily.shape} (ds and y)")
    print(f"  Total days: {len(df_daily)}")
    print(f"  Total sales QTY: {df_daily['y'].sum():,}")

    # Step 3: Train/Test Split
    print("\n[Step 3] Performing Train/Test Split...")
    train_df, test_df = split_train_test(df_daily)
    print(f"  Training set range: {train_df['ds'].min().strftime('%Y-%m-%d')} to {train_df['ds'].max().strftime('%Y-%m-%d')} ({len(train_df)} days)")
    print(f"  Testing set range:  {test_df['ds'].min().strftime('%Y-%m-%d')} to {test_df['ds'].max().strftime('%Y-%m-%d')} ({len(test_df)} days)")

    # Step 4: Fit Prophet on Training Set
    print("\n[Step 4] Fitting Prophet model on training set (with Indian Holidays)...")
    try:
        model = fit_prophet(train_df)
        print("  Prophet model fitted successfully.")
    except Exception as e:
        print(f"\n[ERROR] Fitting model failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 5: Evaluate on Test Set
    print("\n[Step 5] Evaluating model performance on test set (Mar - May 2026)...")
    eval_plot_path = os.path.join(args.output_dir, "evaluation_plot.png")
    rmse, mape, wape = evaluate_model(model, test_df, plot_path=eval_plot_path)
    
    print("-" * 50)
    print(f"  Evaluation Metrics (Daily Sales):")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.2f}")
    if not pd.isna(mape):
        print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}% (calculated on days with non-zero sales)")
    else:
        print("  MAPE: N/A")
    if not pd.isna(wape):
        print(f"  WAPE (Weighted Absolute Percentage Error): {wape:.2f}% (standard for zero-heavy daily sales)")
    print("-" * 50)
    print(f"  Saved evaluation plot to: {eval_plot_path}")

    # Step 6: Refit & Forecast
    # If MAPE/WAPE is acceptable (we will print a warning if WAPE > 25%, but proceed)
    metric_val = wape if not pd.isna(wape) else (mape if not pd.isna(mape) else 0)
    if metric_val > 25.0:
        print(f"\n[WARNING] Error metric ({metric_val:.1f}%) is higher than the desired 20-25% threshold.")
        print("Proceeding with full model refitting anyway...")
    else:
        print(f"\nError metric is within acceptable range ({metric_val:.1f}% <= 25%).")

    print("\n[Step 6] Refitting Prophet model on entire dataset (Oct 2024 - May 2026)...")
    try:
        full_model, forecast_90 = refit_and_forecast(df_daily, forecast_days=90)
        print("  Refit complete. Generated 90-day forecast (Jun 2026 - Aug 2026).")
    except Exception as e:
        print(f"\n[ERROR] Refitting/Forecasting failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Save output forecast
    csv_out_path = os.path.join(args.output_dir, "prophet_90day_forecast.csv")
    forecast_90.to_csv(csv_out_path, index=False)
    print(f"  Saved 90-day forecast to CSV: {csv_out_path}")
    
    # Save components plot (Trend, Holidays, Seasonalities)
    components_plot_path = os.path.join(args.output_dir, "components_plot.png")
    
    # Retrieve full forecast to generate components plot
    future = full_model.make_future_dataframe(periods=90, freq='D')
    forecast_full = full_model.predict(future)
    
    fig = full_model.plot_components(forecast_full)
    fig.savefig(components_plot_path, dpi=300)
    plt.close(fig)
    print(f"  Saved model seasonality/holiday components plot to: {components_plot_path}")

    # Print summary of the forecast
    print("\n" + "=" * 50)
    print("  90-DAY FORECAST SUMMARY (Jun - Aug 2026)")
    print("=" * 50)
    forecast_90_summary = forecast_90.copy()
    forecast_90_summary['Month'] = forecast_90_summary['ds'].dt.to_period('M')
    monthly_summary = forecast_90_summary.groupby('Month').agg(
        Total_Forecasted_QTY=('yhat', 'sum'),
        Min_Forecasted_QTY=('yhat', 'min'),
        Max_Forecasted_QTY=('yhat', 'max'),
        Days_in_Month=('ds', 'count')
    ).reset_index()
    
    for idx, row in monthly_summary.iterrows():
        print(f"  * Month: {row['Month']}")
        print(f"    - Predicted QTY to Sell: {row['Total_Forecasted_QTY']:.0f} units")
        print(f"    - Daily Prediction Range: {row['Min_Forecasted_QTY']:.0f} to {row['Max_Forecasted_QTY']:.0f} units")
        print(f"    - Forecasted Days: {row['Days_in_Month']}")
        print()
    print("=" * 60)
    print("  Execution completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
