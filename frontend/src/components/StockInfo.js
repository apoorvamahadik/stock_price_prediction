import React from 'react';
import { Card, Row, Col, Badge } from 'react-bootstrap';

function StockInfo({ info }) {
  if (!info) return null;

  const formatNumber = (num) => {
    // Handle null/undefined
    if (num == null) return '$0.00';
    
    // Convert to number if it's a string
    let number;
    if (typeof num === 'string') {
      // Remove any formatting characters
      const cleanStr = num.replace(/[^0-9.-]+/g, '');
      number = parseFloat(cleanStr);
      if (isNaN(number)) return '$0.00';
    } else if (typeof num === 'number') {
      number = num;
    } else {
      return '$0.00';
    }
    
    // Format based on size
    if (number >= 1e12) return `$${(number / 1e12).toFixed(2)}T`;
    if (number >= 1e9) return `$${(number / 1e9).toFixed(2)}B`;
    if (number >= 1e6) return `$${(number / 1e6).toFixed(2)}M`;
    if (number >= 1e3) return `$${(number / 1e3).toFixed(2)}K`;
    return `$${number.toFixed(2)}`;
  };

  const formatPrice = (price) => {
    if (price == null) return '$0.00';
    
    let number;
    if (typeof price === 'string') {
      const cleanStr = price.replace(/[^0-9.-]+/g, '');
      number = parseFloat(cleanStr);
      if (isNaN(number)) return '$0.00';
    } else if (typeof price === 'number') {
      number = price;
    } else {
      return '$0.00';
    }
    
    return `$${number.toFixed(2)}`;
  };

  const safeToFixed = (value, decimals = 2) => {
    if (value == null) return 'N/A';
    if (typeof value === 'number') return value.toFixed(decimals);
    if (typeof value === 'string') {
      const num = parseFloat(value);
      return isNaN(num) ? 'N/A' : num.toFixed(decimals);
    }
    return 'N/A';
  };

  return (
    <Card>
      <Card.Header>
        <h5 className="mb-0">📊 Stock Information</h5>
        {info.is_mock_data && (
          <Badge bg="warning" className="mt-2">Demo Data</Badge>
        )}
      </Card.Header>
      <Card.Body>
        <h6 className="mb-3">
          <strong>{info.company_name || 'N/A'}</strong>
        </h6>
        
        <Row className="mb-2">
          <Col xs={6}>
            <small className="text-muted">Current Price</small>
            <div className="fw-bold">{formatPrice(info.current_price)}</div>
          </Col>
          <Col xs={6}>
            <small className="text-muted">Previous Close</small>
            <div className="fw-bold">{formatPrice(info.previous_close)}</div>
          </Col>
        </Row>
        
        <Row className="mb-2">
          <Col xs={6}>
            <small className="text-muted">Market Cap</small>
            <div>{formatNumber(info.market_cap)}</div>
          </Col>
          <Col xs={6}>
            <small className="text-muted">P/E Ratio</small>
            <div>{safeToFixed(info.pe_ratio)}</div>
          </Col>
        </Row>
        
        <Row className="mb-2">
          <Col xs={6}>
            <small className="text-muted">Dividend Yield</small>
            <div>{info.dividend_yield ? `${safeToFixed(info.dividend_yield)}%` : 'N/A'}</div>
          </Col>
          <Col xs={6}>
            <small className="text-muted">Volume</small>
            <div>{formatNumber(info.volume)}</div>
          </Col>
        </Row>
        
        <Row className="mb-2">
          <Col xs={6}>
            <small className="text-muted">Day Range</small>
            <div>{formatPrice(info.day_low)} – {formatPrice(info.day_high)}</div>
          </Col>
          <Col xs={6}>
            <small className="text-muted">52W Range</small>
            <div>{formatPrice(info.fifty_two_week_low)} – {formatPrice(info.fifty_two_week_high)}</div>
          </Col>
        </Row>
        
        {info.volatility && (
          <Row className="mb-2">
            <Col xs={12}>
              <small className="text-muted">Volatility</small>
              <div>{safeToFixed(info.volatility * 100)}%</div>
            </Col>
          </Row>
        )}
        
        {info.data_source && (
          <div className="mt-3 small text-muted">
            <small>Data source: {info.data_source}</small>
            {info.data_points && (
              <span> • {info.data_points} data points</span>
            )}
          </div>
        )}
      </Card.Body>
    </Card>
  );
}

export default StockInfo;