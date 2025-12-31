# quick_test_local.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

def test_existing_model(data_file, model_file="amzn_lstm_model.h5", sequence_length=60):
    """Test a previously trained model"""
    
    # Load data
    df = pd.read_csv(data_file)
    data = df['Close'].values.reshape(-1, 1)
    
    # Load model
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        return
    
    model = load_model(model_file)
    
    # Prepare scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data)
    
    # Prepare test data (last sequence_length days)
    test_data = data[-sequence_length:]
    test_scaled = scaler.transform(test_data)
    test_sequence = test_scaled.reshape(1, sequence_length, 1)
    
    # Make prediction
    prediction_scaled = model.predict(test_sequence, verbose=0)
    prediction = scaler.inverse_transform(prediction_scaled)
    
    print(f"Test data shape: {test_data.shape}")
    print(f"Last actual price: ${data[-1][0]:.2f}")
    print(f"\nNext 18 days predictions:")
    for i, price in enumerate(prediction[0], 1):
        print(f"Day {i}: ${price:.2f}")
    
    return prediction[0]

if __name__ == "__main__":
    # Test with your AMZN data
    data_file = "backend/data/AMZN.csv"
    test_existing_model(data_file)