import React from 'react';
import { Card, Row, Col } from 'react-bootstrap';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Line } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function StockChart({ historical = [], predictions = [] }) {
  // Process data for chart
  const processDataForChart = () => {
    // Combine all data
    const allData = [...historical, ...predictions];
    
    if (allData.length === 0) {
      return {
        labels: [],
        datasets: []
      };
    }

    // Create labels (just sequential numbers for simplicity)
    const labels = allData.map((_, index) => {
      if (index < historical.length) {
        return `Day ${index + 1}`;
      } else {
        return `Prediction ${index - historical.length + 1}`;
      }
    });

    // Create datasets
    const historicalData = historical.map(item => 
      parseFloat(item.Close || item.price || item.Predicted || 0)
    );
    
    const predictionData = predictions.map(item => 
      parseFloat(item.Predicted || item.price || 0)
    );

    // Pad historical data with null for prediction section
    const paddedHistorical = [...historicalData];
    while (paddedHistorical.length < allData.length) {
      paddedHistorical.push(null);
    }

    // Pad prediction data with null for historical section  
    const paddedPredictions = Array(historical.length).fill(null).concat(predictionData);

    return {
      labels,
      datasets: [
        {
          label: 'Historical Prices',
          data: paddedHistorical,
          borderColor: 'rgb(53, 162, 235)',
          backgroundColor: 'rgba(53, 162, 235, 0.5)',
          pointRadius: 2,
          pointHoverRadius: 4,
          tension: 0.1,
          spanGaps: true
        },
        {
          label: 'Predictions',
          data: paddedPredictions,
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
          borderDash: [5, 5],
          pointRadius: 3,
          pointHoverRadius: 5,
          tension: 0.1,
          spanGaps: true
        }
      ]
    };
  };

  const chartData = processDataForChart();

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Stock Price Prediction Chart',
        font: {
          size: 16
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          label: function(context) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              label += new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
              }).format(context.parsed.y);
            }
            return label;
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Time Period'
        },
        grid: {
          display: false
        }
      },
      y: {
        beginAtZero: false,
        title: {
          display: true,
          text: 'Price (USD)'
        },
        ticks: {
          callback: function(value) {
            return '$' + value.toFixed(2);
          }
        }
      }
    },
    interaction: {
      intersect: false,
      mode: 'nearest'
    }
  };

  // Calculate statistics
  const calculateStats = () => {
    if (historical.length === 0 && predictions.length === 0) {
      return null;
    }

    const historicalPrices = historical.map(item => 
      parseFloat(item.Close || item.price || item.Predicted || 0)
    );
    const predictionPrices = predictions.map(item => 
      parseFloat(item.Predicted || item.price || 0)
    );

    const lastHistorical = historicalPrices.length > 0 ? historicalPrices[historicalPrices.length - 1] : 0;
    const firstPrediction = predictionPrices.length > 0 ? predictionPrices[0] : 0;
    const lastPrediction = predictionPrices.length > 0 ? predictionPrices[predictionPrices.length - 1] : 0;

    const totalChange = lastPrediction && lastHistorical ? 
      ((lastPrediction - lastHistorical) / lastHistorical * 100) : 0;

    const avgPrediction = predictionPrices.length > 0 ? 
      predictionPrices.reduce((a, b) => a + b, 0) / predictionPrices.length : 0;

    return {
      historicalCount: historical.length,
      predictionCount: predictions.length,
      lastHistoricalPrice: lastHistorical,
      firstPredictionPrice: firstPrediction,
      lastPredictionPrice: lastPrediction,
      averagePrediction: avgPrediction,
      totalChange: totalChange,
      minHistorical: historicalPrices.length > 0 ? Math.min(...historicalPrices) : 0,
      maxHistorical: historicalPrices.length > 0 ? Math.max(...historicalPrices) : 0,
      minPrediction: predictionPrices.length > 0 ? Math.min(...predictionPrices) : 0,
      maxPrediction: predictionPrices.length > 0 ? Math.max(...predictionPrices) : 0
    };
  };

  const stats = calculateStats();

  return (
    <Card className="mb-4">
      <Card.Header>
        <h5 className="mb-0">📈 Stock Price Prediction</h5>
      </Card.Header>
      <Card.Body>
        {historical.length > 0 || predictions.length > 0 ? (
          <>
            <div style={{ height: '400px' }}>
              <Line options={options} data={chartData} />
            </div>
            
            {stats && (
              <div className="mt-4">
                <Row>
                  <Col md={6}>
                    <div className="card mb-3">
                      <div className="card-body p-3">
                        <h6 className="card-subtitle mb-2 text-muted">Historical Data</h6>
                        <div className="d-flex justify-content-between">
                          <span>Period:</span>
                          <span className="fw-bold">{stats.historicalCount} days</span>
                        </div>
                        <div className="d-flex justify-content-between">
                          <span>Last Price:</span>
                          <span className="fw-bold">${stats.lastHistoricalPrice.toFixed(2)}</span>
                        </div>
                        <div className="d-flex justify-content-between">
                          <span>Range:</span>
                          <span className="fw-bold">
                            ${stats.minHistorical.toFixed(2)} - ${stats.maxHistorical.toFixed(2)}
                          </span>
                        </div>
                      </div>
                    </div>
                  </Col>
                  
                  <Col md={6}>
                    <div className="card mb-3">
                      <div className="card-body p-3">
                        <h6 className="card-subtitle mb-2 text-muted">Prediction</h6>
                        <div className="d-flex justify-content-between">
                          <span>Period:</span>
                          <span className="fw-bold">{stats.predictionCount} days</span>
                        </div>
                        <div className="d-flex justify-content-between">
                          <span>First Predicted:</span>
                          <span className="fw-bold">${stats.firstPredictionPrice.toFixed(2)}</span>
                        </div>
                        <div className="d-flex justify-content-between">
                          <span>Last Predicted:</span>
                          <span className="fw-bold">${stats.lastPredictionPrice.toFixed(2)}</span>
                        </div>
                        <div className="d-flex justify-content-between">
                          <span>Total Change:</span>
                          <span className={`fw-bold ${stats.totalChange >= 0 ? 'text-success' : 'text-danger'}`}>
                            {stats.totalChange >= 0 ? '+' : ''}{stats.totalChange.toFixed(2)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </Col>
                </Row>
                
                <div className="alert alert-info mt-3">
                  <small>
                    <strong>Legend:</strong> Blue line shows historical prices, Red dashed line shows predicted future prices.
                    Predictions are based on machine learning models and should not be considered financial advice.
                  </small>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="text-center py-5">
            <p className="text-muted">No chart data available.</p>
            <p className="small">Select a stock and click "Predict" to generate chart.</p>
          </div>
        )}
      </Card.Body>
    </Card>
  );
}

export default StockChart;