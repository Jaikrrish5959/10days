# Prophet Sales Forecasting Pipeline

This project builds a time-series forecasting model using [Prophet](https://facebook.github.io/prophet/) to predict aggregate daily sales quantities.

## Requirements

Ensure Python 3.8+ is installed. The dependencies are listed in `requirements.txt`:
- `prophet` (for modeling)
- `pandas` (for data manipulation)
- `openpyxl` (for reading Excel files)
- `matplotlib` (for generating plots)
- `scikit-learn` (for computing regression metrics)

## Installation

Create a virtual environment and install the dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Pipeline

To run the pipeline using the default sample dataset path:
```bash
.venv\Scripts\python.exe main.py
```

To run with a custom file path or output directory:
```bash
.venv\Scripts\python.exe main.py --file "path/to/your/sales_data.xlsx" --output-dir "custom_output"
```

## Steps Executed
1. **Load & Clean**: Loads the Excel file, parses invoice dates, filters out negative quantities (returns), and trims dates to the range Oct 1, 2024 to May 31, 2026.
2. **Aggregate & Format**: Groups sales daily, fills any gaps in dates with a sale quantity of `0`, and renames the columns to `ds` and `y`.
3. **Train/Test Split**: Splits the data into a training set (Oct 2024 - Feb 28, 2026) and a test set (Mar 2026 - May 2026).
4. **Model Training**: Fits a Prophet model on the training data with weekly/yearly seasonalities and Indian public holidays.
5. **Evaluation**: Predicts the test set, computes Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and Weighted Absolute Percentage Error (WAPE), and saves a comparison plot.
6. **Refit & Forecast**: Refits on the complete historical dataset and forecasts 90 days into the future (Jun - Aug 2026).

## Outputs Generated
All outputs are saved to the `--output-dir` (default is `output/`):
- `evaluation_plot.png`: Line plot comparing actual vs. predicted daily sales for the test period (Mar - May 2026) with uncertainty intervals.
- `components_plot.png`: Prophet's visualization of the trend, weekly/yearly seasonality, and holiday effects.
- `prophet_90day_forecast.csv`: A CSV file containing the forecasted date (`ds`), point forecast (`yhat`), and lower/upper bounds (`yhat_lower`, `yhat_upper`) for June - August 2026.
