
# Crime Rate Analytics: Enhancing Safety Through Crime Rate Forecasting

A comprehensive data science platform for crime analysis, forecasting, and risk assessment using machine learning and interactive visualizations.

## 🎯 Project Overview

This platform provides advanced analytics for crime data, featuring:

- **Crime Forecasting**: ARIMA, SARIMA, and LSTM models for predicting future crime trends
- **Risk Classification**: Random Forest-based risk assessment for different areas
- **Hotspot Mapping**: Interactive Folium heatmaps for crime pattern visualization
- **Real-time Dashboard**: React-based frontend for data exploration and analysis

## 🏗️ Project Structure

```
crime-rate-analytics/
├── backend/                    # Flask API Backend
│   ├── app.py                 # Main Flask application
│   ├── ml/                    # Machine Learning modules
│   │   ├── forecast_engine.py # Time series forecasting
│   │   ├── risk_classifier.py # Risk classification
│   │   └── hotspot_visualizer.py # Hotspot mapping
│   ├── models/                # Saved ML models
│   ├── datasets/              # Data storage
│   │   └── merged_crime_data.csv
│   ├── static/                # Static files
│   │   ├── forecast_plots/    # Generated forecast plots
│   │   └── hotspot_maps/      # Generated hotspot maps
│   └── requirements.txt       # Python dependencies
│
├── frontend/                  # React Frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── ForecastPlots.jsx
│   │   │   ├── HotspotMap.jsx
│   │   │   └── RiskPredictionForm.jsx
│   │   ├── pages/            # Page components
│   │   │   ├── Dashboard.jsx
│   │   │   └── Login.jsx
│   │   └── api.js            # API utilities
│
├── scripts/                   # Utility scripts
│   ├── data_preprocessing.py  # Data preprocessing
│   └── retrain_models.py     # Model retraining
│
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd crime-rate-analytics/backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application:**
   ```bash
   python app.py
   ```

   The API will be available at `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start development server:**
   ```bash
   npm start
   ```

   The frontend will be available at `http://localhost:3000`

## 📊 Features

### 1. Crime Forecasting
- **Models**: ARIMA, SARIMA, LSTM
- **Capabilities**: District-level monthly forecasting
- **Evaluation**: Automatic model comparison using MAE/MSE
- **Output**: Interactive plots and forecast data

### 2. Risk Classification
- **Algorithm**: Random Forest Classifier
- **Features**: Crime statistics, temporal patterns, geographic factors
- **Output**: Risk levels (High/Medium/Low) with confidence scores

### 3. Hotspot Visualization
- **Technology**: Folium interactive maps
- **Features**: Heatmaps, district markers, risk indicators
- **Interactivity**: Zoom, pan, popup information

### 4. Dashboard Interface
- **Framework**: React with Tailwind CSS
- **Features**: Tabbed interface, real-time updates, data export
- **Responsive**: Mobile and desktop compatible

## 🔧 API Endpoints

### Health Check
```
GET /health
```

### Dataset Information
```
GET /dataset-info
```

### Crime Forecasting
```
POST /forecast-district
{
  "district": "string",
  "months_ahead": 6
}
```

### Risk Prediction
```
POST /predict-risk
{
  "features": {
    "total_crimes": 100,
    "crime_rate": 10.0,
    ...
  }
}
```

### Hotspot Mapping
```
GET /hotspot-map
```

## 🛠️ Data Processing

### Preprocessing Script
```bash
cd scripts
python data_preprocessing.py --input ../backend/datasets/merged_crime_data.csv --output ../backend/datasets/
```

### Model Retraining
```bash
cd scripts
python retrain_models.py --dataset ../backend/datasets/merged_crime_data.csv --model-type all
```

## 📈 Machine Learning Models

### Time Series Forecasting

1. **ARIMA (AutoRegressive Integrated Moving Average)**
   - Best for: Linear trends and patterns
   - Parameters: Auto-selected based on data

2. **SARIMA (Seasonal ARIMA)**
   - Best for: Seasonal patterns
   - Parameters: Includes seasonal components

3. **LSTM (Long Short-Term Memory)**
   - Best for: Complex non-linear patterns
   - Architecture: 2-layer LSTM with dropout

### Risk Classification

- **Algorithm**: Random Forest with 100 estimators
- **Features**: 12 crime-related features
- **Evaluation**: Cross-validation with accuracy metrics

## 🎨 Frontend Components

### Dashboard
- Multi-tab interface for different analytics
- Real-time data updates
- Export functionality

### ForecastPlots
- Interactive plot selection
- Model comparison tables
- Download options

### HotspotMap
- Embedded Folium maps
- Fullscreen mode
- Legend and instructions

### RiskPredictionForm
- Dynamic feature input
- Real-time prediction
- Feature importance visualization

## 🔒 Security Features

- Input validation and sanitization
- Error handling and logging
- CORS configuration
- Authentication ready (demo mode included)

## 📝 Configuration

### Environment Variables
```bash
# Backend
FLASK_ENV=development
FLASK_DEBUG=True

# Frontend
REACT_APP_API_URL=http://localhost:5000
```

### Model Parameters
- Forecast lookback: 12 months
- LSTM epochs: 50
- Random Forest estimators: 100

## 🧪 Testing

### Backend Testing
```bash
cd backend
python -m pytest tests/
```

### Frontend Testing
```bash
cd frontend
npm test
```

## 📦 Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Production Considerations
- Use production WSGI server (Gunicorn)
- Configure reverse proxy (Nginx)
- Set up SSL certificates
- Use production database
- Enable monitoring and logging

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Crime data sources and providers
- Open source ML libraries (scikit-learn, TensorFlow, statsmodels)
- Visualization libraries (Folium, Matplotlib, React)
- Community contributors and feedback

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Contact the development team
- Check documentation and examples

---

**Crime Rate Analytics Platform** - Enhancing Safety Through Data Science 🛡️

