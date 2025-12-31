import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const stockAPI = {
  // Health check
  healthCheck: () => api.get('/api/health'),
  
  // Get predictions
  getPredictions: (ticker, days = 30, model = 'lstm') => 
    api.post('/api/predict', { ticker, days, model }),
  
  // Get stock statistics
  getStockStats: (ticker) => 
    api.get(`/api/stats/${ticker}`),
  
  // Get historical data
  getHistoricalData: (ticker, days = 365) => 
    api.get(`/api/historical/${ticker}?days=${days}`),
  
  // Get available models
  getModels: () => 
    api.get('/api/models'),
  
  // Batch predictions for comparison
  getModelComparison: async (ticker, days = 30) => {
    const [lstmResult, arimaResult] = await Promise.all([
      api.post('/api/predict', { ticker, days, model: 'lstm' }),
      api.post('/api/predict', { ticker, days, model: 'arima' })
    ]);
    
    return {
      lstm: lstmResult.data,
      arima: arimaResult.data
    };
  },
  
  // Search stocks
  searchStocks: async (query) => {
    // This would typically connect to a search API
    // For now, we'll simulate with popular stocks
    const popularStocks = [
      { symbol: 'AAPL', name: 'Apple Inc.' },
      { symbol: 'MSFT', name: 'Microsoft Corporation' },
      { symbol: 'GOOGL', name: 'Alphabet Inc.' },
      { symbol: 'AMZN', name: 'Amazon.com Inc.' },
      { symbol: 'TSLA', name: 'Tesla Inc.' },
      { symbol: 'META', name: 'Meta Platforms Inc.' },
      { symbol: 'NVDA', name: 'NVIDIA Corporation' },
      { symbol: 'JPM', name: 'JPMorgan Chase & Co.' },
      { symbol: 'V', name: 'Visa Inc.' },
      { symbol: 'JNJ', name: 'Johnson & Johnson' }
    ];
    
    return popularStocks.filter(stock => 
      stock.symbol.toLowerCase().includes(query.toLowerCase()) ||
      stock.name.toLowerCase().includes(query.toLowerCase())
    );
  }
};

export default api;