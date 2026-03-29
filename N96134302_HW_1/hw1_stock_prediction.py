import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def main():
    # 1. Download Data
    print("Downloading S&P 500 data...")
    ticker = "^GSPC"
    # yfinance 'end' parameter is exclusive, so we use 2026-01-01 to include all of 2025-12-31
    start_date = "2021-01-01"
    end_date = "2026-01-01" 
    
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Handle MultiIndex columns if returning from newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    # We focus on the closing price
    df = df[['Close']].copy() 
    df.index = pd.to_datetime(df.index)
    
    # 2. Feature Engineering
    print("Engineering features (Lags and Moving Averages)...")
    # Lag features: capture the price from previous days
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_3'] = df['Close'].shift(3)
    df['Lag_5'] = df['Close'].shift(5)
    
    # Moving averages: smooth out price trends over 5 and 20 days
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    
    # Drop missing values caused by shifting and rolling at the beginning of the dataset
    df.dropna(inplace=True)
    
    # Define features and target variable
    features = ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'MA_5', 'MA_20']
    target = 'Close'
    
    X = df[features]
    y = df[target]
    
    # 3. Train-Test Split (Sequential split for time-series data, without random shuffle)
    print("Splitting data into training (2021-2024) and testing (2025)...")
    train_mask = (df.index >= "2021-01-01") & (df.index <= "2024-12-31")
    test_mask = (df.index >= "2025-01-01") & (df.index <= "2025-12-31")
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # 4. Model Training
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    print("Training XGBoost...")
    xgb_model = XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
    xgb_model.fit(X_train, y_train)
    
    # 5. Predictions
    rf_predictions = rf_model.predict(X_test)
    xgb_predictions = xgb_model.predict(X_test)
    
    # 6. Evaluation
    rf_mse = mean_squared_error(y_test, rf_predictions)
    xgb_mse = mean_squared_error(y_test, xgb_predictions)
    
    print(f"Random Forest MSE: {rf_mse:.2f}")
    print(f"XGBoost MSE: {xgb_mse:.2f}")
    
    # Save MSE results to file
    with open("mse_result.txt", "w") as f:
        f.write(f"Random Forest MSE: {rf_mse:.2f}\n")
        f.write(f"XGBoost MSE: {xgb_mse:.2f}\n")
        
    # 7. Visualization
    print("Plotting actual vs predicted prices...")
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test.values, label='Actual Price', color='black', linewidth=2)
    plt.plot(y_test.index, rf_predictions, label='Random Forest Prediction', color='blue', linestyle='--', alpha=0.7)
    plt.plot(y_test.index, xgb_predictions, label='XGBoost Prediction', color='red', linestyle='-.', alpha=0.7)
    
    plt.title('S&P 500 (^GSPC) Stock Price Prediction (2025 Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    # Save plot to file
    plot_filename = "stock_prediction_results.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_filename}")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()
