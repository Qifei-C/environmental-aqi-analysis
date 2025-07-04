# Environmental Air Quality Index (AQI) Analysis

A comprehensive toolkit for analyzing air quality data, including data collection, processing, visualization, and predictive modeling of environmental air quality metrics.

## Overview

This project provides tools for environmental air quality analysis, including:
- Air quality data collection from various sources (NOAA, EPA, OpenWeather)
- Data preprocessing and cleaning
- Statistical analysis and trend detection
- Air quality prediction modeling
- Interactive visualizations and reporting

## Features

- **Multi-source Data Integration**: Collect data from NOAA, EPA, and weather APIs
- **Real-time Monitoring**: Support for real-time AQI monitoring and alerts
- **Predictive Modeling**: Machine learning models for AQI forecasting
- **Interactive Dashboards**: Web-based visualization dashboards
- **Statistical Analysis**: Comprehensive statistical analysis of air quality trends
- **Geospatial Analysis**: Location-based air quality mapping and analysis

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.data_pipeline import AQIDataPipeline
from src.analysis import AQIAnalyzer
from src.visualization import AQIVisualizer

# Initialize data pipeline
pipeline = AQIDataPipeline()

# Load and process data
data = pipeline.load_data('data/aqi_data.csv')
processed_data = pipeline.preprocess_data(data)

# Analyze air quality trends
analyzer = AQIAnalyzer()
trends = analyzer.analyze_trends(processed_data)
predictions = analyzer.predict_aqi(processed_data)

# Create visualizations
visualizer = AQIVisualizer()
visualizer.plot_aqi_trends(processed_data)
visualizer.create_dashboard(processed_data)
```

### Data Collection

```python
from src.data_collectors import NOAACollector, EPACollector

# Collect data from NOAA
noaa_collector = NOAACollector(api_token='your_noaa_token')
noaa_data = noaa_collector.collect_data(
    start_date='2023-01-01',
    end_date='2023-12-31',
    location='40.7128,-74.0060'  # New York City
)

# Collect data from EPA
epa_collector = EPACollector()
epa_data = epa_collector.collect_data(
    state_code='NY',
    county_code='061'
)
```

## Configuration

### API Setup

You'll need API tokens for data collection:

1. **NOAA API Token**: Register at https://www.ncdc.noaa.gov/cdo-web/token
2. **EPA API**: No token required for basic access
3. **OpenWeather API**: Register at https://openweathermap.org/api

Create a `.env` file:
```
NOAA_API_TOKEN=your_noaa_token_here
OPENWEATHER_API_KEY=your_openweather_key_here
EPA_API_ENDPOINT=https://aqs.epa.gov/data/api
```

### Configuration File

Edit `config/config.yaml`:
```yaml
data_sources:
  noaa:
    enabled: true
    parameters:
      dataset: "GHCND"
      datatypes: ["TMAX", "TMIN", "PRCP"]
  
  epa:
    enabled: true
    parameters:
      pollutants: ["PM25", "PM10", "O3", "NO2", "SO2", "CO"]
  
analysis:
  time_series:
    seasonal_decomposition: true
    trend_analysis: true
  
  prediction:
    models: ["ARIMA", "Random Forest", "LSTM"]
    forecast_horizon: 30  # days

visualization:
  dashboard:
    port: 8050
    debug: false
```

## Data Sources

### Supported Data Sources

1. **NOAA Climate Data**
   - Temperature, precipitation, humidity
   - Historical weather patterns
   - Climate station data

2. **EPA Air Quality System (AQS)**
   - PM2.5, PM10, Ozone, NO2, SO2, CO measurements
   - Air quality index calculations
   - Regulatory monitoring data

3. **OpenWeather API**
   - Real-time air pollution data
   - Weather forecast data
   - UV index information

### Data Format

Expected data columns:
- `date`: Date of measurement (YYYY-MM-DD)
- `location`: Location identifier or coordinates
- `aqi`: Air Quality Index value
- `pm25`: PM2.5 concentration (μg/m³)
- `pm10`: PM10 concentration (μg/m³)
- `o3`: Ozone concentration (ppm)
- `no2`: NO2 concentration (ppm)
- `so2`: SO2 concentration (ppm)
- `co`: CO concentration (ppm)
- `temperature`: Temperature (°C)
- `humidity`: Relative humidity (%)
- `wind_speed`: Wind speed (m/s)

## Analysis Features

### Statistical Analysis
- Trend analysis and seasonality detection
- Correlation analysis between pollutants
- Anomaly detection
- Time series decomposition

### Predictive Modeling
- ARIMA models for time series forecasting
- Machine learning models (Random Forest, XGBoost)
- Deep learning models (LSTM, GRU)
- Ensemble forecasting

### Geospatial Analysis
- Spatial interpolation of air quality data
- Hotspot analysis
- Geographic clustering
- Distance-based analysis

## Visualization

### Available Plots
- Time series plots with trend lines
- Correlation heatmaps
- Geographic maps with AQI overlays
- Distribution plots and histograms
- Interactive dashboards

### Dashboard Features
- Real-time data updates
- Interactive filtering and selection
- Multi-location comparison
- Export functionality

## Project Structure

```
environmental-aqi-analysis/
├── data/                   # Data files
│   ├── raw/               # Raw data files
│   ├── processed/         # Processed data
│   └── external/          # External data sources
├── src/                   # Source code
│   ├── data_pipeline.py   # Main data processing pipeline
│   ├── data_collectors.py # Data collection modules
│   ├── analysis.py        # Analysis functions
│   ├── models.py          # Prediction models
│   ├── visualization.py   # Visualization tools
│   └── utils.py           # Utility functions
├── config/                # Configuration files
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── requirements.txt
```

## API Reference

### AQIDataPipeline
Main class for data processing and analysis pipeline.

### AQIAnalyzer
Statistical analysis and trend detection methods.

### AQIPredictor
Machine learning models for AQI prediction.

### AQIVisualizer
Visualization and dashboard creation tools.

## Performance

### Benchmarks
- Data processing: ~1M records/minute
- Model training: 2-5 minutes for daily forecasts
- Dashboard loading: <3 seconds for 1-year data

### Optimization
- Efficient pandas operations for large datasets
- Caching for repeated API calls
- Async data collection for multiple sources
- Optimized visualization rendering

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NOAA for climate data access
- EPA for air quality monitoring data
- OpenWeather for real-time air quality APIs
- Scientific community for air quality research