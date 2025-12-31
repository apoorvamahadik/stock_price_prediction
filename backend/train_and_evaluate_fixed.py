# train_and_evaluate_fixed.py
import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend
matplotlib.use('Agg')  # This prevents blocking
import matplotlib.pyplot as plt
import warnings
import os
import sys

warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

def main():
    print("\n" + "="*70)
    print("🎯 COMPLETE AMAZON STOCK PRICE PREDICTION")
    print("="*70)
    
    # ============================================================================
    # 1. LOAD DATA
    # ============================================================================
    print("\n📊 STEP 1: LOADING DATA")
    print("-"*40)
    
    filepath = "data/AMZN.csv"
    print(f"Loading data from: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Successfully loaded {len(df)} rows")
        
        print(f"\nDataset Information:")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df.iloc[0, 0]} to {df.iloc[-1, 0]}")
        
        # Use Close prices
        data = df['Close'].values.reshape(-1, 1)
        
        print(f"\n💰 Price Statistics:")
        print(f"   Min: ${data.min():.2f}")
        print(f"   Max: ${data.max():.2f}")
        print(f"   Current: ${data[-1][0]:.2f}")
        print(f"   Mean: ${data.mean():.2f}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # ============================================================================
    # 2. PREPARE DATA
    # ============================================================================
    print(f"\n📊 STEP 2: PREPARING DATA")
    print("-"*40)
    
    sequence_length = 60
    prediction_days = 18
    train_ratio = 0.8
    
    print(f"Sequence length: {sequence_length} days")
    print(f"Prediction horizon: {prediction_days} days")
    print(f"Train/Test split: {train_ratio*100:.0f}%/{100-train_ratio*100:.0f}%")
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length - prediction_days):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length:i + sequence_length + prediction_days])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, prediction_days)
    
    print(f"\nCreated {len(X)} sequences")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Split data
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nData Split:")
    print(f"Training: {len(X_train)} sequences")
    print(f"Testing: {len(X_test)} sequences")
    
    # ============================================================================
    # 3. BUILD MODEL
    # ============================================================================
    print(f"\n🤖 STEP 3: BUILDING LSTM MODEL")
    print("-"*40)
    
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(prediction_days)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # ============================================================================
    # 4. TRAIN MODEL
    # ============================================================================
    print(f"\n🚀 STEP 4: TRAINING MODEL")
    print("-"*40)
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # ============================================================================
    # 5. EVALUATE MODEL
    # ============================================================================
    print(f"\n📊 STEP 5: EVALUATING MODEL")
    print("-"*40)
    
    # Make predictions on test set
    print("Making predictions on test set...")
    y_pred_scaled = model.predict(X_test, verbose=1)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test)
    
    # Flatten for metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / y_true_flat)) * 100
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # Calculate directional accuracy
    if len(y_true_flat) > 1:
        true_direction = np.sign(np.diff(y_true_flat))
        pred_direction = np.sign(np.diff(y_pred_flat))
        dir_accuracy = np.mean(true_direction == pred_direction) * 100
    else:
        dir_accuracy = 0
    
    print(f"\n{'='*60}")
    print("📈 PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"Mean Absolute Error (MAE):      ${mae:.2f}")
    print(f"Root Mean Square Error (RMSE):  ${rmse:.2f}")
    print(f"Mean Absolute % Error (MAPE):   {mape:.2f}%")
    print(f"R² Score:                       {r2:.4f}")
    print(f"Directional Accuracy:           {dir_accuracy:.1f}%")
    
    # Interpret results
    print(f"\n📊 ACCURACY ASSESSMENT:")
    print(f"{'-'*40}")
    
    if mape < 5:
        accuracy = "✅ EXCELLENT"
        assessment = "Very accurate predictions - Model is highly reliable"
    elif mape < 10:
        accuracy = "👍 GOOD" 
        assessment = "Acceptable for stock predictions - Model is reliable"
    elif mape < 20:
        accuracy = "⚠️  MODERATE"
        assessment = "Room for improvement - Use with caution"
    else:
        accuracy = "❌ POOR"
        assessment = "Needs significant improvement - Not reliable"
    
    print(f"Overall Accuracy: {accuracy}")
    print(f"Assessment: {assessment}")
    
    if dir_accuracy > 60:
        print(f"Trading Signals: 📈 Good ({dir_accuracy:.1f}% accuracy)")
    elif dir_accuracy > 50:
        print(f"Trading Signals: ↔️  Better than random ({dir_accuracy:.1f}% accuracy)")
    else:
        print(f"Trading Signals: 📉 Poor ({dir_accuracy:.1f}% accuracy)")
    
    # ============================================================================
    # 6. MAKE FUTURE PREDICTIONS
    # ============================================================================
    print(f"\n🔮 STEP 6: FUTURE PREDICTIONS (Next 18 Days)")
    print("-"*40)
    
    # Get last sequence
    last_sequence = data[-sequence_length:]
    last_scaled = scaler.transform(last_sequence)
    last_scaled = last_scaled.reshape(1, sequence_length, 1)
    
    # Predict future
    future_scaled = model.predict(last_scaled, verbose=1)
    future_prices = scaler.inverse_transform(future_scaled)[0]
    
    current_price = data[-1][0]
    
    print(f"\n💰 Current AMZN Price: ${current_price:.2f}")
    print(f"📅 Using last {sequence_length} days to predict next {prediction_days} days")
    print(f"\n{'='*70}")
    print(f"{'Day':<6} {'Predicted Price':<15} {'Change ($)':<15} {'Change (%)':<12}")
    print(f"{'-'*70}")
    
    for i, price in enumerate(future_prices, 1):
        change_dollar = price - current_price
        change_percent = (change_dollar / current_price) * 100
        sign = "+" if change_dollar >= 0 else ""
        print(f"Day {i:<3} ${price:<14.2f} {sign}{change_dollar:<14.2f} {sign}{change_percent:<11.2f}%")
    
    print(f"{'-'*70}")
    
    # Summary statistics
    min_pred = np.min(future_prices)
    max_pred = np.max(future_prices)
    avg_pred = np.mean(future_prices)
    std_pred = np.std(future_prices)
    
    print(f"\n📊 PREDICTION SUMMARY:")
    print(f"Minimum predicted price:  ${min_pred:.2f}")
    print(f"Maximum predicted price:  ${max_pred:.2f}")
    print(f"Average predicted price:  ${avg_pred:.2f}")
    print(f"Prediction volatility:    ${std_pred:.2f} (std dev)")
    
    overall_change = ((avg_pred - current_price) / current_price) * 100
    
    print(f"\n📈 TREND ANALYSIS:")
    print(f"Expected price change:    {overall_change:+.2f}%")
    
    if overall_change > 10:
        trend = "🚀 STRONGLY BULLISH"
    elif overall_change > 5:
        trend = "📈 BULLISH"
    elif overall_change > 0:
        trend = "↗️  SLIGHTLY BULLISH"
    elif overall_change > -5:
        trend = "↘️  SLIGHTLY BEARISH"
    elif overall_change > -10:
        trend = "📉 BEARISH"
    else:
        trend = "🐻 STRONGLY BEARISH"
    
    print(f"Overall trend:            {trend}")
    
    # ============================================================================
    # 7. SAVE MODEL
    # ============================================================================
    print(f"\n💾 STEP 7: SAVING MODEL")
    print("-"*40)
    
    # Save model
    model.save("amzn_lstm_predictor.h5")
    print("✅ Model saved as 'amzn_lstm_predictor.h5'")
    
    # Save scaler
    joblib.dump(scaler, "amzn_scaler.pkl")
    print("✅ Scaler saved as 'amzn_scaler.pkl'")
    
    # Save metrics
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'dir_accuracy': dir_accuracy,
        'current_price': current_price,
        'avg_predicted': avg_pred,
        'predicted_change': overall_change
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("amzn_predictor_metrics.csv", index=False)
    print("✅ Metrics saved as 'amzn_predictor_metrics.csv'")
    
    # ============================================================================
    # 8. FINAL SUMMARY
    # ============================================================================
    print(f"\n{'='*70}")
    print("🎯 FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n📈 Model Performance:")
    print(f"   MAPE: {mape:.2f}% - {accuracy}")
    print(f"   R² Score: {r2:.4f}")
    print(f"   Directional Accuracy: {dir_accuracy:.1f}%")
    
    print(f"\n🔮 Future Predictions:")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   Average Prediction (next 18 days): ${avg_pred:.2f}")
    print(f"   Expected Change: {overall_change:+.2f}%")
    print(f"   Trend: {trend}")
    
    print(f"\n💡 Recommendations:")
    if mape < 10:
        print("   ✅ Model is accurate enough for investment analysis")
        print("   ✅ Can be used for medium-term trend forecasting")
        print("   ✅ Combine with fundamental analysis for best results")
    else:
        print("   ⚠️  Model needs improvement for reliable predictions")
        print("   ⚠️  Use with caution for financial decisions")
        print("   ⚠️  Consider adding more features or data")
    
    print(f"\n{'='*70}")
    print("✅ TRAINING AND EVALUATION COMPLETE!")
    print(f"{'='*70}")
    
    # Now generate plots (they won't block)
    print(f"\n📊 Generating evaluation plots...")
    
    # Plot 1: Training History
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss During Training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['mae'], label='Training MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    ax2.set_title('Model MAE During Training')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
    print("✅ Saved: training_history.png")
    
    # Plot 2: Actual vs Predicted
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Actual vs Predicted
    sample_indices = np.arange(min(200, len(y_true_flat)))
    axes[0, 0].plot(sample_indices, y_true_flat[:len(sample_indices)], label='Actual', linewidth=2, color='blue')
    axes[0, 0].plot(sample_indices, y_pred_flat[:len(sample_indices)], label='Predicted', linestyle='--', linewidth=2, color='red')
    axes[0, 0].set_title('Actual vs Predicted Prices')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    axes[0, 1].scatter(y_true_flat, y_pred_flat, alpha=0.6, s=10, color='green')
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 1].set_title('Actual vs Predicted (Scatter)')
    axes[0, 1].set_xlabel('Actual Price ($)')
    axes[0, 1].set_ylabel('Predicted Price ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    errors = y_true_flat - y_pred_flat
    axes[1, 0].hist(errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Prediction Error Distribution')
    axes[1, 0].set_xlabel('Error ($) [Actual - Predicted]')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Cumulative error
    cumulative_error = np.cumsum(np.abs(errors))
    axes[1, 1].plot(cumulative_error, linewidth=2, color='purple')
    axes[1, 1].set_title('Cumulative Absolute Error')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Cumulative Error ($)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=100, bbox_inches='tight')
    print("✅ Saved: evaluation_results.png")
    
    # Plot 3: Future Predictions
    plt.figure(figsize=(12, 6))
    
    # Plot historical data (last 120 days)
    historical_days = 120
    historical_prices = data[-historical_days:].flatten()
    historical_indices = np.arange(-historical_days, 0)
    
    # Plot future predictions
    future_indices = np.arange(0, prediction_days)
    
    plt.plot(historical_indices, historical_prices, label='Historical', linewidth=2, color='blue')
    plt.plot([-1, 0], [historical_prices[-1], current_price], 'blue', linewidth=2)
    plt.plot([0], [current_price], 'ro', markersize=10, label='Current Price')
    plt.plot(future_indices, future_prices, label='Predicted', linewidth=2, color='red', linestyle='--')
    plt.fill_between(future_indices, min_pred, max_pred, alpha=0.2, color='red', label='Prediction Range')
    
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.title(f'Amazon Stock Price Prediction\nCurrent: ${current_price:.2f} | Forecast: ${avg_pred:.2f} ({overall_change:+.1f}%)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Days (0 = Today)', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('future_predictions.png', dpi=100, bbox_inches='tight')
    print("✅ Saved: future_predictions.png")
    
    print(f"\n📁 All files saved successfully!")
    print(f"   - amzn_lstm_predictor.h5 (model)")
    print(f"   - amzn_scaler.pkl (scaler)")
    print(f"   - amzn_predictor_metrics.csv (metrics)")
    print(f"   - training_history.png (training plots)")
    print(f"   - evaluation_results.png (evaluation plots)")
    print(f"   - future_predictions.png (future forecast)")

if __name__ == "__main__":
    main()