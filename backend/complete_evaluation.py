# complete_evaluation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def continue_evaluation():
    """Continue evaluation from where the training left off"""
    
    print(f"\n{'='*70}")
    print("📊 CONTINUING MODEL EVALUATION")
    print(f"{'='*70}")
    
    # We need to recreate the predictor or load it
    # Since you're in the same session, let's check if variables exist
    try:
        # Check if predictor exists
        if 'predictor' not in globals():
            print("❌ Predictor object not found. Creating new one...")
            
            # Recreate from scratch
            from train_test_local import StockPredictor
            predictor = StockPredictor(sequence_length=60, prediction_days=18)
            
            # Load data
            data = predictor.load_data("data/AMZN.csv")
            
            # Prepare data
            X_train, X_test, y_train, y_test = predictor.prepare_data()
            
            # Build and load model (if saved)
            try:
                from tensorflow.keras.models import load_model
                import joblib
                predictor.model = load_model("amzn_lstm_predictor.h5")
                predictor.scaler = joblib.load("amzn_lstm_predictor_scaler.pkl")
                print("✅ Loaded saved model")
            except:
                print("❌ No saved model found. Please train first.")
                return
        else:
            print("✅ Found existing predictor object")
            
        # Now evaluate
        print("\n1. Evaluating on test set...")
        metrics, y_pred, y_true = predictor.evaluate()
        
        print(f"\n2. Making future predictions...")
        future_prices = predictor.predict_future()
        
        print(f"\n{'='*70}")
        print("📋 FINAL ACCURACY ASSESSMENT")
        print(f"{'='*70}")
        
        # Detailed accuracy analysis
        print(f"\n🔍 ACCURACY ANALYSIS:")
        print(f"   MAPE: {metrics['mape']:.2f}% - ", end="")
        if metrics['mape'] < 5:
            print("✅ EXCELLENT accuracy (MAPE < 5%)")
            print("   This model is very accurate for stock predictions")
        elif metrics['mape'] < 10:
            print("👍 GOOD accuracy (5% ≤ MAPE < 10%)")
            print("   This is acceptable accuracy for stock price predictions")
        elif metrics['mape'] < 20:
            print("⚠️  MODERATE accuracy (10% ≤ MAPE < 20%)")
            print("   Consider improving the model")
        else:
            print("❌ POOR accuracy (MAPE ≥ 20%)")
            print("   Model needs significant improvement")
        
        print(f"\n   R² Score: {metrics['r2']:.4f} - ", end="")
        if metrics['r2'] > 0.9:
            print("✅ Excellent fit (explains >90% of variance)")
        elif metrics['r2'] > 0.7:
            print("👍 Good fit (explains >70% of variance)")
        elif metrics['r2'] > 0.5:
            print("⚠️  Moderate fit (explains >50% of variance)")
        else:
            print("❌ Poor fit (explains <50% of variance)")
        
        print(f"\n   Directional Accuracy: {metrics['dir_accuracy']:.1f}% - ", end="")
        if metrics['dir_accuracy'] > 60:
            print("📈 Good for trading signals")
        elif metrics['dir_accuracy'] > 50:
            print("↔️  Better than random guessing")
        else:
            print("📉 Worse than random guessing")
        
        # Check for common prediction issues
        print(f"\n🔧 MODEL DIAGNOSTICS:")
        
        # Check error distribution
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        errors = y_true_flat - y_pred_flat
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        print(f"   Mean error: ${mean_error:.2f} (should be close to 0)")
        print(f"   Error std: ${std_error:.2f} (lower is better)")
        
        if abs(mean_error) > 5:
            print("   ⚠️  Bias detected: Predictions are systematically off")
        
        # Check prediction range
        pred_range = np.ptp(y_pred_flat)  # Peak-to-peak (max-min)
        actual_range = np.ptp(y_true_flat)
        
        print(f"\n   Predicted price range: ${pred_range:.2f}")
        print(f"   Actual price range: ${actual_range:.2f}")
        
        if pred_range < actual_range * 0.5:
            print("   ⚠️  Under-predicting volatility")
        elif pred_range > actual_range * 1.5:
            print("   ⚠️  Over-predicting volatility")
        
        # Future prediction analysis
        current_price = predictor.data[-1][0]
        avg_future = np.mean(future_prices)
        change_percent = ((avg_future - current_price) / current_price) * 100
        
        print(f"\n🔮 FUTURE PREDICTION ANALYSIS:")
        print(f"   Current price: ${current_price:.2f}")
        print(f"   Average predicted (next 18 days): ${avg_future:.2f}")
        print(f"   Expected change: {change_percent:+.2f}%")
        
        # Confidence interval
        pred_std = np.std(future_prices)
        confidence_95_low = avg_future - 1.96 * pred_std
        confidence_95_high = avg_future + 1.96 * pred_std
        
        print(f"\n   95% Confidence Interval:")
        print(f"   Lower bound: ${confidence_95_low:.2f}")
        print(f"   Upper bound: ${confidence_95_high:.2f}")
        print(f"   Range: ${confidence_95_high - confidence_95_low:.2f}")
        
        # Risk assessment
        print(f"\n📊 RISK ASSESSMENT:")
        if pred_std / avg_future > 0.05:
            print("   ⚠️  High volatility expected")
        elif pred_std / avg_future > 0.02:
            print("   📈 Moderate volatility expected")
        else:
            print("   📉 Low volatility expected")
        
        print(f"\n{'='*70}")
        print("💡 RECOMMENDATIONS:")
        
        if metrics['mape'] < 10:
            print("1. ✅ Model is accurate enough for trend analysis")
            print("2. ✅ Can be used for medium-term predictions")
            print("3. ✅ Consider using in combination with fundamental analysis")
        else:
            print("1. ⚠️  Model needs improvement for reliable predictions")
            print("2. ⚠️  Use with caution for decision making")
            print("3. ⚠️  Consider adding more features or data")
        
        print(f"\n{'='*70}")
        print("🎯 EVALUATION COMPLETE!")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    continue_evaluation()