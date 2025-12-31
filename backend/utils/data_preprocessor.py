import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        logger.info("DataPreprocessor initialized")
    
    def prepare_data(self, df):
        """Prepare and clean stock data"""
        try:
            df = df.copy()
            
            # Standardize column names
            df.columns = [col.strip().title() for col in df.columns]
            
            # Ensure Date column exists and is datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            elif 'Datetime' in df.columns:
                df['Date'] = pd.to_datetime(df['Datetime'])
                df = df.drop('Datetime', axis=1, errors='ignore')
            else:
                # Create date range if no date column
                df['Date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='D')
            
            # Ensure Close column exists
            if 'Close' not in df.columns:
                if 'Adj Close' in df.columns:
                    df['Close'] = df['Adj Close']
                elif 'Price' in df.columns:
                    df['Close'] = df['Price']
                else:
                    # Try to find numeric column
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        df['Close'] = df[numeric_cols[0]]
                    else:
                        raise ValueError("No numeric column found for prices")
            
            # Sort by date
            df = df.sort_values('Date')
            
            # Fill missing values
            df['Close'] = df['Close'].fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"Data prepared: {len(df)} rows, columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            logger.error(f"Data preprocessing error: {e}")
            raise