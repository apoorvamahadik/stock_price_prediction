# Create a simple test script in backend/test_models.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models.lstm_model import StockLSTMModel
from models.arima_model import ARIMAModel

# Create sample data
dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
prices = 100 + np.cumsum(np.random.randn(100) * 2)
df = pd.DataFrame({'Date': dates, 'Close': prices})

print(f"Sample data created: {len(df)} rows")
print(f"Last price: ${df['Close'].iloc[-1]:.2f}")

# Test LSTM model
print("\n=== Testing LSTM Model ===")
lstm = StockLSTMModel()
lstm_pred = lstm.predict(df, days=10)
print(f"LSTM predictions shape: {lstm_pred.shape}")
print(f"First LSTM prediction: ${lstm_pred['Predicted'].iloc[0]:.2f}")
print(f"Last LSTM prediction: ${lstm_pred['Predicted'].iloc[-1]:.2f}")

# Test ARIMA model
print("\n=== Testing ARIMA Model ===")
arima = ARIMAModel()
arima_pred = arima.predict(df, days=10)
print(f"ARIMA predictions shape: {arima_pred.shape}")
print(f"First ARIMA prediction: ${arima_pred['Predicted'].iloc[0]:.2f}")
print(f"Last ARIMA prediction: ${arima_pred['Predicted'].iloc[-1]:.2f}")