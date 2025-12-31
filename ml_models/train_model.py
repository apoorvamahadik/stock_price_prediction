import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pickle

def train_and_save_model(ticker='AMZN', lookback=60, epochs=50):
    """Train LSTM model and save it"""
    
    # Create ml_models directory if it doesn't exist
    os.makedirs('ml_models', exist_ok=True)
    
    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # 3 years of data
    
    print(f"Fetching data for {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if stock_data.empty:
        print(f"No data found for {ticker}")
        return
    
    # Prepare data
    df = stock_data.reset_index()
    data = df[['Close']].values
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Build model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model
    print("Training model...")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=1)
    
    # Save model
    model_path = 'ml_models/saved_lstm_model.h5'
    scaler_path = 'ml_models/scaler.pkl'
    
    model.save(model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    
    # Test prediction
    test_input = scaled_data[-lookback:].reshape(1, lookback, 1)
    prediction = model.predict(test_input, verbose=0)
    prediction_price = scaler.inverse_transform(prediction)[0][0]
    
    print(f"Sample prediction for next day: ${prediction_price:.2f}")
    
    return model, scaler

if __name__ == '__main__':
    # Train model for Amazon
    train_and_save_model(ticker='AMZN', epochs=30)