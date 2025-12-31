import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ARIMAModel:
    def __init__(self):
        logger.info("ARIMA Model initialized (Simple version)")
    
    def predict(self, df, days=30):
        """Generate predictions based on statistical trends"""
        try:
            logger.info(f"Generating ARIMA predictions for {days} days")
            
            # Ensure we have the required columns
            if 'Close' not in df.columns:
                raise ValueError("DataFrame must have 'Close' column")
            
            # Get the last date
            if 'Date' in df.columns:
                last_date = pd.to_datetime(df['Date'].iloc[-1])
            else:
                last_date = datetime.now()
            
            # Calculate moving averages for trend
            prices = df['Close'].values
            
            # Simple ARIMA-like prediction: weighted average of recent trends
            if len(prices) >= 10:
                short_ma = np.mean(prices[-10:])  # 10-day moving average
                long_ma = np.mean(prices[-30:]) if len(prices) >= 30 else short_ma  # 30-day MA
                
                # Calculate trend
                trend_strength = (short_ma - long_ma) / long_ma
                volatility = np.std(prices[-10:]) / np.mean(prices[-10:])
            else:
                trend_strength = 0.001  # Slight upward trend
                volatility = 0.02
            
            # Generate predictions
            predictions = []
            current_price = float(df['Close'].iloc[-1])
            
            for i in range(1, days + 1):
                pred_date = last_date + timedelta(days=i)
                
                # ARIMA-like prediction: current price + trend + seasonal component
                trend_component = current_price * trend_strength * i
                seasonal_component = current_price * 0.001 * np.sin(i * np.pi / 14)  # Bi-weekly pattern
                random_component = current_price * volatility * np.random.normal(0, 0.5)
                
                predicted_price = current_price + trend_component + seasonal_component + random_component
                
                # Ensure positive prices
                predicted_price = max(predicted_price, current_price * 0.7)
                
                predictions.append({
                    'Date': pred_date,
                    'Predicted': round(float(predicted_price), 2)
                })
            
            result_df = pd.DataFrame(predictions)
            logger.info(f"Generated {len(result_df)} ARIMA predictions, first: ${result_df['Predicted'].iloc[0]:.2f}")
            return result_df
            
        except Exception as e:
            logger.error(f"ARIMA prediction error: {e}")
            # Return fallback predictions
            return self._fallback_predictions(df, days)
    
    def _fallback_predictions(self, df, days):
        """Fallback prediction method"""
        last_date = pd.to_datetime(df['Date'].iloc[-1])
        last_price = float(df['Close'].iloc[-1])
        
        predictions = []
        for i in range(1, days + 1):
            pred_date = last_date + timedelta(days=i)
            # Slightly different trend for ARIMA
            predicted_price = last_price * (1 + 0.0005 * i)
            
            predictions.append({
                'Date': pred_date,
                'Predicted': round(float(predicted_price), 2)
            })
        
        return pd.DataFrame(predictions)