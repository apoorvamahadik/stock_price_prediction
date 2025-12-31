# train_test_local.py - CLEAN VERSION
import numpy as np
import pandas as pd
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

class StockPredictor:
    def __init__(self, sequence_length=60, prediction_days=18):
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.data = None
        self.df = None
        self.history = None
        
    def load_data(self, filepath="data/AMZN.csv"):
        """Load stock data from CSV file"""
        print(f"🔍 Looking for data at: {filepath}")
        print(f"📂 Current directory: {os.getcwd()}")
        
        # Convert to absolute path
        abs_path = os.path.abspath(filepath)
        print(f"📁 Absolute path: {abs_path}")
        
        if not os.path.exists(filepath):
            # Try to find the file
            print("\n🔎 Searching for file...")
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if 'AMZN' in file and file.endswith('.csv'):
                        found_path = os.path.join(root, file)
                        print(f"✅ Found: {found_path}")
                        filepath = found_path
                        break
        
        print(f"\n📊 Loading data from: {filepath}")
        
        try:
            # Load CSV
            self.df = pd.read_csv(filepath)
            print(f"✅ Successfully loaded {len(self.df)} rows")
            
            # Display info
            print(f"\n📈 Dataset Information:")
            print(f"   Shape: {self.df.shape}")
            print(f"   Columns: {list(self.df.columns)}")
            print(f"   Date range: {self.df.iloc[0, 0]} to {self.df.iloc[-1, 0]}")
            
            # Check for Close column
            if 'Close' in self.df.columns:
                self.data = self.df['Close'].values.reshape(-1, 1)
                print(f"\n💰 Price Statistics:")
                print(f"   Min: ${self.data.min():.2f}")
                print(f"   Max: ${self.data.max():.2f}")
                print(f"   Current: ${self.data[-1][0]:.2f}")
                print(f"   Mean: ${self.data.mean():.2f}")
            else:
                print("\n⚠️  Warning: 'Close' column not found!")
                print("   Available columns:", list(self.df.columns))
                # Use first numeric column
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    self.data = self.df[col].values.reshape(-1, 1)
                    print(f"   Using '{col}' column instead")
                else:
                    raise ValueError("No numeric columns found in dataset")
            
            return self.data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def prepare_data(self, train_ratio=0.8):
        """Prepare and split the data"""
        print(f"\n{'='*60}")
        print("📊 DATA PREPARATION")
        print(f"{'='*60}")
        
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print(f"Total data points: {len(self.data)}")
        print(f"Sequence length: {self.sequence_length} days")
        print(f"Prediction horizon: {self.prediction_days} days")
        
        # Check if we have enough data
        min_required = self.sequence_length + self.prediction_days + 10
        if len(self.data) < min_required:
            raise ValueError(f"Not enough data. Need at least {min_required} points, have {len(self.data)}")
        
        # Scale data
        scaled_data = self.scaler.fit_transform(self.data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length - self.prediction_days):
            X.append(scaled_data[i:i + self.sequence_length])
            y.append(scaled_data[i + self.sequence_length:i + self.sequence_length + self.prediction_days])
        
        X = np.array(X)
        y = np.array(y).reshape(-1, self.prediction_days)
        
        print(f"\nCreated {len(X)} sequences")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        
        # Split data
        split_idx = int(len(X) * train_ratio)
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]
        
        print(f"\nData Split:")
        print(f"Training: {len(self.X_train)} sequences ({train_ratio*100:.0f}%)")
        print(f"Testing: {len(self.X_test)} sequences ({100-train_ratio*100:.0f}%)")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model(self):
        """Build the LSTM model"""
        print(f"\n{'='*60}")
        print("🤖 BUILDING LSTM MODEL")
        print(f"{'='*60}")
        
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(self.prediction_days)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),  # Use legacy for M1/M2
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        model.summary()
        return model
    
    def train(self, epochs=50, batch_size=32):
        """Train the model"""
        print(f"\n{'='*60}")
        print("🚀 TRAINING MODEL")
        print(f"{'='*60}")
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training progress"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss During Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE During Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate(self):
        """Evaluate model performance"""
        print(f"\n{'='*70}")
        print("📈 MODEL EVALUATION ON TEST SET")
        print(f"{'='*70}")
        
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        # Make predictions
        print("Making predictions on test set...")
        y_pred_scaled = self.model.predict(self.X_test, verbose=1)
        y_pred = self.scaler.inverse_transform(y_pred_scaled)
        y_true = self.scaler.inverse_transform(self.y_test)
        
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
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"{'-'*60}")
        print(f"Mean Absolute Error (MAE):      ${mae:.2f}")
        print(f"Root Mean Square Error (RMSE):  ${rmse:.2f}")
        print(f"Mean Absolute % Error (MAPE):   {mape:.2f}%")
        print(f"R² Score:                       {r2:.4f}")
        print(f"Directional Accuracy:           {dir_accuracy:.1f}%")
        
        # Interpret results
        print(f"\n📈 INTERPRETATION:")
        print(f"{'-'*60}")
        
        if mape < 5:
            print("✅ EXCELLENT! MAPE < 5% - Very accurate predictions")
        elif mape < 10:
            print("👍 GOOD! 5% ≤ MAPE < 10% - Acceptable for stock predictions")
        elif mape < 20:
            print("⚠️  MODERATE! 10% ≤ MAPE < 20% - Room for improvement")
        else:
            print("❌ POOR! MAPE ≥ 20% - Model needs significant improvement")
        
        if dir_accuracy > 60:
            print("📈 Good for trading signals (>60% accuracy)")
        elif dir_accuracy > 50:
            print("↔️  Better than random guessing (>50% accuracy)")
        else:
            print("📉 Worse than random guessing")
        
        # Plot predictions vs actual
        self.plot_evaluation_results(y_true, y_pred, y_true_flat, y_pred_flat)
        
        return {
            'mae': mae, 'rmse': rmse, 'mape': mape,
            'r2': r2, 'dir_accuracy': dir_accuracy
        }, y_pred, y_true
    
    def plot_evaluation_results(self, y_true, y_pred, y_true_flat, y_pred_flat):
        """Plot evaluation results"""
        print(f"\n{'='*70}")
        print("📊 VISUALIZING PREDICTIONS")
        print(f"{'='*70}")
        
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Actual vs Predicted
        ax1 = plt.subplot(2, 2, 1)
        sample_indices = np.arange(min(100, len(y_true_flat)))
        ax1.plot(sample_indices, y_true_flat[:len(sample_indices)], label='Actual', linewidth=2, color='blue')
        ax1.plot(sample_indices, y_pred_flat[:len(sample_indices)], label='Predicted', linestyle='--', linewidth=2, color='red')
        ax1.set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sample Index', fontsize=12)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. Scatter plot
        ax2 = plt.subplot(2, 2, 2)
        ax2.scatter(y_true_flat, y_pred_flat, alpha=0.6, s=20, color='green')
        min_val = min(y_true_flat.min(), y_pred_flat.min())
        max_val = max(y_true_flat.max(), y_pred_flat.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax2.set_title('Actual vs Predicted (Scatter)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Actual Price ($)', fontsize=12)
        ax2.set_ylabel('Predicted Price ($)', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 3. Error distribution
        ax3 = plt.subplot(2, 2, 3)
        errors = y_true_flat - y_pred_flat
        ax3.hist(errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax3.axvline(x=np.mean(errors), color='blue', linestyle='-', linewidth=2, label=f'Mean: ${np.mean(errors):.2f}')
        ax3.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Error ($) [Actual - Predicted]', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative error
        ax4 = plt.subplot(2, 2, 4)
        cumulative_error = np.cumsum(np.abs(errors))
        ax4.plot(cumulative_error, linewidth=2, color='purple')
        ax4.set_title('Cumulative Absolute Error', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Sample Index', fontsize=12)
        ax4.set_ylabel('Cumulative Error ($)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Model Prediction Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    def predict_future(self):
        """Make future predictions"""
        print(f"\n{'='*70}")
        print("🔮 FUTURE PREDICTIONS (Next 18 Days)")
        print(f"{'='*70}")
        
        if self.data is None:
            raise ValueError("No data loaded")
        
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        # Get last sequence
        last_sequence = self.data[-self.sequence_length:]
        last_scaled = self.scaler.transform(last_sequence)
        last_scaled = last_scaled.reshape(1, self.sequence_length, 1)
        
        # Predict
        print("Making future predictions...")
        future_scaled = self.model.predict(last_scaled, verbose=1)
        future_prices = self.scaler.inverse_transform(future_scaled)[0]
        
        current_price = self.data[-1][0]
        
        print(f"\n💰 Current AMZN Price: ${current_price:.2f}")
        print(f"📅 Using last {self.sequence_length} days to predict next {self.prediction_days} days")
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
        price_range = ((max_pred - min_pred) / avg_pred) * 100
        
        print(f"\n📈 TREND ANALYSIS:")
        print(f"Expected price change:    {overall_change:+.2f}%")
        print(f"Predicted price range:    ±{price_range/2:.1f}% around average")
        
        if overall_change > 10:
            trend = "🚀 STRONGLY BULLISH"
            confidence = "High"
        elif overall_change > 5:
            trend = "📈 BULLISH"
            confidence = "Medium"
        elif overall_change > 0:
            trend = "↗️  SLIGHTLY BULLISH"
            confidence = "Low"
        elif overall_change > -5:
            trend = "↘️  SLIGHTLY BEARISH"
            confidence = "Low"
        elif overall_change > -10:
            trend = "📉 BEARISH"
            confidence = "Medium"
        else:
            trend = "🐻 STRONGLY BEARISH"
            confidence = "High"
        
        print(f"Overall trend:            {trend}")
        print(f"Confidence:               {confidence}")
        
        # Visualize future predictions
        self.plot_future_predictions(future_prices, current_price)
        
        return future_prices
    
    def plot_future_predictions(self, future_prices, current_price):
        """Plot future predictions"""
        plt.figure(figsize=(12, 6))
        
        # Plot historical data (last 120 days)
        historical_days = 120
        historical_prices = self.data[-historical_days:].flatten()
        historical_indices = np.arange(-historical_days, 0)
        
        # Plot future predictions
        future_indices = np.arange(0, self.prediction_days)
        
        plt.plot(historical_indices, historical_prices, label='Historical', linewidth=2, color='blue')
        plt.plot([-1, 0], [historical_prices[-1], current_price], 'blue', linewidth=2)
        plt.plot([0], [current_price], 'ro', markersize=10, label='Current Price')
        plt.plot(future_indices, future_prices, label='Predicted', linewidth=2, color='red', linestyle='--')
        plt.fill_between(future_indices, np.min(future_prices), np.max(future_prices), 
                        alpha=0.2, color='red', label='Prediction Range')
        
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        plt.title(f'Amazon Stock Price Prediction\nCurrent: ${current_price:.2f} | Forecast: ${np.mean(future_prices):.2f} ({((np.mean(future_prices) - current_price)/current_price*100):+.1f}%)', 
                fontsize=14, fontweight='bold')
        plt.xlabel('Days (0 = Today)', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_name="amzn_lstm_predictor"):
        """Save the trained model and scaler"""
        if self.model is None:
            print("❌ No model to save. Train the model first.")
            return
        
        # Save model
        self.model.save(f"{model_name}.h5")
        print(f"✅ Model saved as '{model_name}.h5'")
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, f"{model_name}_scaler.pkl")
        print(f"✅ Scaler saved as '{model_name}_scaler.pkl'")
        
        # Save metrics if available
        if hasattr(self, 'metrics'):
            metrics_df = pd.DataFrame([self.metrics])
            metrics_df.to_csv(f"{model_name}_metrics.csv", index=False)
            print(f"✅ Metrics saved as '{model_name}_metrics.csv'")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("🎯 AMAZON STOCK PRICE PREDICTOR")
    print("="*70)
    
    # Initialize predictor
    predictor = StockPredictor(
        sequence_length=60,
        prediction_days=18
    )
    
    try:
        # Load data
        data = predictor.load_data("data/AMZN.csv")
        
        # Prepare data
        X_train, X_test, y_train, y_test = predictor.prepare_data()
        
        # Build model
        predictor.build_model()
        
        # Train model
        history = predictor.train(epochs=50)
        
        # Plot training history
        predictor.plot_training_history()
        
        # Evaluate model
        metrics, y_pred, y_true = predictor.evaluate()
        predictor.metrics = metrics  # Store metrics for saving
        
        # Make future predictions
        future_prices = predictor.predict_future()
        
        # Save model
        print(f"\n{'='*70}")
        save_choice = input("💾 Do you want to save the trained model? (y/n): ").strip().lower()
        if save_choice == 'y':
            model_name = input("Enter model name (or press Enter for 'amzn_lstm_predictor'): ").strip()
            if not model_name:
                model_name = "amzn_lstm_predictor"
            predictor.save_model(model_name)
        
        print(f"\n{'='*70}")
        print("🎉 TRAINING AND EVALUATION COMPLETE!")
        print("="*70)
        
        # Print final summary
        print(f"\n📋 FINAL SUMMARY:")
        print(f"✅ Model trained successfully")
        print(f"✅ MAPE: {metrics['mape']:.2f}%")
        print(f"✅ R² Score: {metrics['r2']:.4f}")
        print(f"✅ Directional Accuracy: {metrics['dir_accuracy']:.1f}%")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()