import React, { useState, useEffect } from 'react';
import StockChart from './components/StockChart';
import PredictionForm from './components/PredictionForm';
import StockInfo from './components/StockInfo';
import { Container, Row, Col, Alert, Spinner } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import './styles/App.css';

// API base URL - CHANGE THIS TO YOUR BACKEND URL
const API_BASE_URL = 'http://localhost:5001';

function App() {
  const [stockData, setStockData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [stockInfo, setStockInfo] = useState(null);
  const [availableStocks, setAvailableStocks] = useState(['AMZN', 'AAPL', 'GOOGL', 'TSLA', 'MSFT']);
  const [availableModels, setAvailableModels] = useState([
    { id: 'lstm', name: 'LSTM Neural Network' },
    { id: 'arima', name: 'ARIMA Model' }
  ]);

  // Fetch available models on mount
  useEffect(() => {
    fetchAvailableModels();
    // Load default stock (Amazon)
    handlePrediction('AMZN', 30, 'lstm');
  }, []);

  const fetchAvailableModels = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/models`);
      if (response.ok) {
        const data = await response.json();
        setAvailableModels(data.models || availableModels);
      }
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const handlePrediction = async (ticker, days, model) => {
    setLoading(true);
    setError(null);
    setStockInfo(null);
    
    try {
      // Fetch stock info
      const infoResponse = await fetch(`${API_BASE_URL}/api/stats/${ticker}`);
      if (!infoResponse.ok) {
        throw new Error(`API Error: ${infoResponse.status}`);
      }
      const infoData = await infoResponse.json();
      
      if (infoData.error) {
        setError(infoData.error);
      } else {
        setStockInfo(infoData);
      }
      
      // Make prediction request
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ticker: ticker.toUpperCase(),
          days: parseInt(days),
          model: model
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Prediction API Error: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
      } else {
        // Ensure data is in correct format
        const historical = Array.isArray(data.historical) ? data.historical : [];
        const predictions = Array.isArray(data.predictions) ? data.predictions : [];
        
        // Process data for chart
        const processedHistorical = historical.map(item => ({
          ...item,
          date: item.Date || item.date,
          price: item.Close || item.price || item.Predicted || 0
        }));
        
        const processedPredictions = predictions.map(item => ({
          ...item,
          date: item.Date || item.date,
          price: item.Predicted || item.price || 0,
          isPrediction: true
        }));
        
        setStockData(processedHistorical);
        setPredictions(processedPredictions);
      }
    } catch (err) {
      console.error('Prediction error details:', err);
      
      // Provide more helpful error messages
      if (err.message.includes('Failed to fetch')) {
        setError('Cannot connect to the backend server. Make sure the backend is running on http://localhost:5001');
      } else if (err.message.includes('429')) {
        setError('Rate limited. Please wait a moment and try again.');
      } else if (err.message.includes('500')) {
        setError('Server error. Please check if the CSV data file exists in the backend/data directory.');
      } else {
        setError(err.message || 'Failed to fetch predictions. Please try again.');
      }
      
      // Set mock data for development
      setMockData(ticker, days);
    } finally {
      setLoading(false);
    }
  };

  // Fallback mock data function
  const setMockData = (ticker, days) => {
    // Generate mock historical data
    const mockHistorical = [];
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 100);
    
    let basePrice = 150 + Math.random() * 100;
    
    for (let i = 0; i < 100; i++) {
      const date = new Date(startDate);
      date.setDate(startDate.getDate() + i);
      
      basePrice = basePrice * (1 + (Math.random() - 0.5) * 0.02);
      
      mockHistorical.push({
        date: date.toISOString().split('T')[0],
        price: parseFloat(basePrice.toFixed(2)),
        isPrediction: false
      });
    }
    
    // Generate mock predictions
    const mockPredictions = [];
    let lastPrice = mockHistorical[mockHistorical.length - 1].price;
    
    for (let i = 1; i <= days; i++) {
      const date = new Date();
      date.setDate(date.getDate() + i);
      
      lastPrice = lastPrice * (1 + 0.001 * i);
      
      mockPredictions.push({
        date: date.toISOString().split('T')[0],
        price: parseFloat(lastPrice.toFixed(2)),
        isPrediction: true
      });
    }
    
    // Set mock stock info
    setStockInfo({
      company_name: `${ticker} Corporation`,
      current_price: mockHistorical[mockHistorical.length - 1].price,
      previous_close: mockHistorical[mockHistorical.length - 2]?.price || mockHistorical[mockHistorical.length - 1].price,
      market_cap: 1500000000000,
      pe_ratio: 28.5,
      dividend_yield: 0,
      volume: 45000000,
      avg_volume: 40000000,
      day_high: mockHistorical[mockHistorical.length - 1].price * 1.01,
      day_low: mockHistorical[mockHistorical.length - 1].price * 0.99,
      fifty_two_week_high: Math.max(...mockHistorical.map(h => h.price)),
      fifty_two_week_low: Math.min(...mockHistorical.map(h => h.price)),
      currency: 'USD',
      is_mock_data: true
    });
    
    setStockData(mockHistorical);
    setPredictions(mockPredictions);
  };

  return (
    <Container fluid className="app-container">
      <header className="py-4 mb-4 border-bottom">
        <h1 className="text-center">📈 Stock Price Predictor</h1>
        <p className="text-center text-muted">
          Predict future stock prices using LSTM and ARIMA models
        </p>
      </header>

      <Row>
        <Col lg={3} className="sidebar">
          <PredictionForm 
            onSubmit={handlePrediction} 
            availableStocks={availableStocks}
            availableModels={availableModels}
          />
          {stockInfo && <StockInfo info={stockInfo} />}
        </Col>

        <Col lg={9} className="main-content">
          {error && (
            <Alert variant="warning">
              <Alert.Heading>Note</Alert.Heading>
              <p>{error}</p>
              <hr />
              <p className="mb-0">
                Using mock data for demonstration. To use real data:
                <ol>
                  <li>Ensure backend is running: <code>python app.py</code></li>
                  <li>Place CSV files in <code>backend/data/</code> directory</li>
                  <li>Restart both frontend and backend</li>
                </ol>
              </p>
            </Alert>
          )}
          
          {loading ? (
            <div className="text-center py-5">
              <Spinner animation="border" variant="primary" />
              <p className="mt-3">Loading predictions...</p>
            </div>
          ) : (stockData.length > 0 || predictions.length > 0) ? (
            <StockChart 
              historical={stockData} 
              predictions={predictions} 
            />
          ) : !loading ? (
            <div className="text-center py-5">
              <Alert variant="info">
                No data available. Try selecting a stock and clicking "Predict".
              </Alert>
            </div>
          ) : null}
          
          {/* Debug info */}
          <div className="mt-4 p-3 bg-light rounded">
            <small className="text-muted">
              <strong>Debug Info:</strong><br />
              Backend URL: {API_BASE_URL}<br />
              Historical data points: {stockData.length}<br />
              Prediction points: {predictions.length}<br />
              Using {stockInfo?.is_mock_data ? 'mock' : 'real'} data
            </small>
          </div>
        </Col>
      </Row>

      <footer className="mt-5 py-3 border-top text-center text-muted">
        <p>Stock Price Prediction System © 2024 | Using Machine Learning Models</p>
      </footer>
    </Container>
  );
}

export default App;