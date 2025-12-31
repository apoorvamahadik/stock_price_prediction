import React, { useState } from 'react';
import { Card, Form, Button, Row, Col } from 'react-bootstrap';

function PredictionForm({ onSubmit, availableStocks = [], availableModels = [] }) {
  const [ticker, setTicker] = useState('AMZN');
  const [days, setDays] = useState(30);
  const [model, setModel] = useState('lstm');
  const [customTicker, setCustomTicker] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    const selectedTicker = ticker === 'custom' ? customTicker.toUpperCase() : ticker;
    onSubmit(selectedTicker, days, model);
  };

  return (
    <Card className="mb-4">
      <Card.Header>
        <h5 className="mb-0">🔮 Prediction Settings</h5>
      </Card.Header>
      <Card.Body>
        <Form onSubmit={handleSubmit}>
          <Form.Group className="mb-3">
            <Form.Label>Stock Symbol</Form.Label>
            <Row>
              <Col md={8}>
                <Form.Select 
                  value={ticker} 
                  onChange={(e) => setTicker(e.target.value)}
                >
                  {availableStocks.map((stock) => (
                    <option key={stock} value={stock}>
                      {stock}
                    </option>
                  ))}
                  <option value="custom">Custom Ticker</option>
                </Form.Select>
              </Col>
              <Col md={4}>
                {ticker === 'custom' && (
                  <Form.Control
                    type="text"
                    placeholder="e.g., AAPL"
                    value={customTicker}
                    onChange={(e) => setCustomTicker(e.target.value)}
                    required
                  />
                )}
              </Col>
            </Row>
            <Form.Text className="text-muted">
              Enter a stock ticker symbol
            </Form.Text>
          </Form.Group>

          <Form.Group className="mb-3">
            <Form.Label>Prediction Model</Form.Label>
            <Form.Select 
              value={model} 
              onChange={(e) => setModel(e.target.value)}
            >
              {availableModels.map((modelOption) => (
                <option key={modelOption.id} value={modelOption.id}>
                  {modelOption.name}
                </option>
              ))}
            </Form.Select>
            <Form.Text className="text-muted">
              Choose prediction algorithm
            </Form.Text>
          </Form.Group>

          <Form.Group className="mb-3">
            <Form.Label>Prediction Days: {days}</Form.Label>
            <Form.Range
              min="7"
              max="60"
              value={days}
              onChange={(e) => setDays(e.target.value)}
            />
            <Form.Text className="text-muted">
              Predict next {days} trading days
            </Form.Text>
          </Form.Group>

          <Button variant="primary" type="submit" className="w-100">
            Predict Stock Price
          </Button>
          
          <div className="mt-3 small text-muted">
            <p className="mb-1">
              Predictions are based on historical data and machine learning models.
            </p>
            {availableStocks.length === 0 && (
              <p className="mb-0 text-warning">
                ⚠️ Using default stocks. Add CSV files to backend/data for more.
              </p>
            )}
          </div>
        </Form>
      </Card.Body>
    </Card>
  );
}

export default PredictionForm;