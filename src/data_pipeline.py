"""
Environmental Air Quality Analysis Data Pipeline

This module provides comprehensive tools for analyzing air quality data
from EPA, NOAA weather data, and traffic information to understand
environmental impacts on public health.
"""

import os
import re
import requests
import zipfile
import time
import warnings
from urllib.parse import urljoin
from datetime import datetime
from glob import glob
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns

# Optional geospatial imports
try:
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.wkt import loads as wkt_loads
    from shapely.errors import WKTReadingError
    GEOPANDAS_AVAILABLE = True
except ImportError:
    print("WARNING: GeoPandas not found. Geospatial operations will be skipped.")
    print("Install using: pip install geopandas shapely")
    GEOPANDAS_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')


class EnvironmentalDataPipeline:
    """Comprehensive pipeline for environmental data processing and analysis."""
    
    def __init__(self, base_dir: str = None, state_code: int = 36):
        """
        Initialize the environmental data pipeline.
        
        Args:
            base_dir: Base directory for data storage
            state_code: FIPS code for target state (default: 36 for NY)
        """
        self.base_dir = base_dir or os.getcwd()
        self.state_code = state_code
        self.state_name = self._get_state_name(state_code)
        
        # Setup directories
        self.dirs = self._setup_directories()
        
        # Configuration
        self.epa_start_year = 2016
        self.current_year = datetime.now().year
        self.weather_end_year = self.current_year - 1
        
        # API configuration
        self.noaa_token = os.getenv("NOAA_API_TOKEN")
        if not self.noaa_token:
            print("WARNING: NOAA_API_TOKEN not found in environment variables")
            print("Set your token: export NOAA_API_TOKEN='your_token_here'")
        
        # URLs
        self.epa_base_url = "https://aqs.epa.gov/aqsweb/airdata/"
        self.noaa_api_base = "https://www.ncei.noaa.gov/cdo-web/api/v2/"
        
        # Clustering parameters
        self.dbscan_eps_km = 10
        self.dbscan_min_samples = 1
    
    def _get_state_name(self, state_code: int) -> str:
        """Get state abbreviation from FIPS code."""
        state_mapping = {
            36: "NY", 6: "CA", 48: "TX", 12: "FL", 17: "IL",
            42: "PA", 39: "OH", 26: "MI", 13: "GA", 37: "NC"
        }
        return state_mapping.get(state_code, f"STATE_{state_code}")
    
    def _setup_directories(self) -> Dict[str, str]:
        """Create and return directory structure."""
        dirs = {
            'downloads_epa': os.path.join(self.base_dir, "downloads_epa"),
            'downloads_weather': os.path.join(self.base_dir, "downloads_weather"),
            'raw_data': os.path.join(self.base_dir, "raw_data"),
            'processed_data': os.path.join(self.base_dir, "processed_data"),
            'final_output': os.path.join(self.base_dir, "final_output"),
            'visualizations': os.path.join(self.base_dir, "visualizations"),
        }
        
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        return dirs
    
    def download_file(self, url: str, save_dir: str, 
                     filename: Optional[str] = None,
                     skip_existing: bool = True) -> Optional[str]:
        """Download a file from URL with error handling."""
        os.makedirs(save_dir, exist_ok=True)
        
        if not filename:
            filename = url.split("/")[-1]
        
        local_filename = os.path.join(save_dir, filename)
        
        if skip_existing and os.path.exists(local_filename):
            print(f"File exists, skipping: {filename}")
            return local_filename
        
        print(f"Downloading: {url}")
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded: {filename}")
            return local_filename
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            if os.path.exists(local_filename):
                os.remove(local_filename)
            return None
    
    def fetch_epa_pollutant_data(self, pollutant_codes: List[str]) -> pd.DataFrame:
        """
        Download and process EPA air quality data for specified pollutants.
        
        Args:
            pollutant_codes: List of EPA pollutant parameter codes
                           (e.g., ['88101', '44201', '42401', '42101', '42602'])
        
        Returns:
            Combined DataFrame with all pollutant data
        """
        print(f"Fetching EPA data for pollutants: {pollutant_codes}")
        all_data = []
        
        for year in range(self.epa_start_year, self.current_year):
            for code in pollutant_codes:
                filename = f"daily_{code}_{year}.zip"
                url = self.epa_base_url + filename
                
                # Download
                zip_path = self.download_file(
                    url, 
                    self.dirs['downloads_epa'],
                    filename
                )
                
                if zip_path:
                    # Extract and read
                    csv_path = self._extract_and_read_zip(
                        zip_path,
                        self.dirs['raw_data']
                    )
                    
                    if csv_path:
                        try:
                            df = pd.read_csv(csv_path)
                            # Filter by state
                            df = df[df['State Code'] == self.state_code]
                            
                            if not df.empty:
                                df['Year'] = year
                                df['Pollutant_Code'] = code
                                all_data.append(df)
                                print(f"  Loaded {len(df)} records for {code} ({year})")
                        
                        except Exception as e:
                            print(f"  Error reading {csv_path}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Total EPA records loaded: {len(combined_df)}")
            return combined_df
        else:
            print("No EPA data loaded")
            return pd.DataFrame()
    
    def _extract_and_read_zip(self, zip_path: str, extract_dir: str) -> Optional[str]:
        """Extract zip file and return path to first CSV file."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                files = zip_ref.namelist()
                csv_files = [f for f in files if f.endswith('.csv')]
                
                if csv_files:
                    zip_ref.extract(csv_files[0], extract_dir)
                    return os.path.join(extract_dir, csv_files[0])
        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")
        return None
    
    def fetch_weather_data(self, data_types: List[str] = None) -> pd.DataFrame:
        """
        Fetch NOAA weather data for the specified state.
        
        Args:
            data_types: Weather data types to fetch
                       (default: ['TMAX', 'TMIN', 'PRCP', 'AWND'])
        
        Returns:
            DataFrame with weather data
        """
        if not self.noaa_token:
            print("ERROR: NOAA API token required for weather data")
            return pd.DataFrame()
        
        if data_types is None:
            data_types = ['TMAX', 'TMIN', 'PRCP', 'AWND']
        
        print(f"Fetching weather data for {self.state_name} ({self.epa_start_year}-{self.weather_end_year})")
        
        headers = {"token": self.noaa_token}
        all_weather_data = []
        
        for year in range(self.epa_start_year, self.weather_end_year + 1):
            params = {
                "datasetid": "GHCND",
                "locationid": f"FIPS:{str(self.state_code).zfill(2)}",
                "startdate": f"{year}-01-01",
                "enddate": f"{year}-12-31",
                "datatypeid": ",".join(data_types),
                "limit": 1000,
                "units": "standard"
            }
            
            try:
                response = requests.get(
                    self.noaa_api_base + "data",
                    headers=headers,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                if 'results' in data:
                    df = pd.DataFrame(data['results'])
                    all_weather_data.append(df)
                    print(f"  Fetched weather data for {year}: {len(df)} records")
                else:
                    print(f"  No weather data for {year}")
                
                # Rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                print(f"  Error fetching weather data for {year}: {e}")
        
        if all_weather_data:
            combined_df = pd.concat(all_weather_data, ignore_index=True)
            print(f"Total weather records: {len(combined_df)}")
            return combined_df
        else:
            return pd.DataFrame()
    
    def cluster_monitoring_stations(self, df: pd.DataFrame, 
                                  lat_col: str = 'Latitude',
                                  lon_col: str = 'Longitude') -> pd.DataFrame:
        """
        Cluster monitoring stations using DBSCAN for regional analysis.
        
        Args:
            df: DataFrame with station coordinates
            lat_col: Latitude column name
            lon_col: Longitude column name
        
        Returns:
            DataFrame with cluster assignments
        """
        print("Clustering monitoring stations...")
        
        # Prepare coordinates
        coords = df[[lat_col, lon_col]].dropna()
        if coords.empty:
            print("No valid coordinates found")
            return df
        
        # Convert to radians for haversine distance
        coords_rad = np.radians(coords.values)
        
        # Earth radius in km
        earth_radius_km = 6371
        eps_rad = self.dbscan_eps_km / earth_radius_km
        
        # Perform clustering
        clustering = DBSCAN(
            eps=eps_rad,
            min_samples=self.dbscan_min_samples,
            metric='haversine'
        )
        
        cluster_labels = clustering.fit_predict(coords_rad)
        
        # Add cluster information
        df_clustered = df.copy()
        df_clustered['cluster_id'] = -1  # Default for missing coordinates
        df_clustered.loc[coords.index, 'cluster_id'] = cluster_labels
        
        # Calculate cluster centers
        valid_clusters = df_clustered[df_clustered['cluster_id'] >= 0]
        if not valid_clusters.empty:
            cluster_centers = valid_clusters.groupby('cluster_id')[[lat_col, lon_col]].mean()
            
            # Add cluster center coordinates
            for cluster_id, center in cluster_centers.iterrows():
                mask = df_clustered['cluster_id'] == cluster_id
                df_clustered.loc[mask, f'{lat_col}_cluster'] = center[lat_col]
                df_clustered.loc[mask, f'{lon_col}_cluster'] = center[lon_col]
        
        n_clusters = len(df_clustered['cluster_id'].unique()) - (1 if -1 in df_clustered['cluster_id'].unique() else 0)
        print(f"Created {n_clusters} station clusters")
        
        return df_clustered
    
    def aggregate_by_cluster_and_date(self, df: pd.DataFrame, 
                                    value_cols: List[str],
                                    date_col: str = 'Date Local') -> pd.DataFrame:
        """
        Aggregate environmental data by cluster and date.
        
        Args:
            df: DataFrame with clustered data
            value_cols: Columns to aggregate (e.g., pollutant concentrations)
            date_col: Date column name
        
        Returns:
            Aggregated DataFrame
        """
        print("Aggregating data by cluster and date...")
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Filter valid clusters and dates
        valid_data = df[(df['cluster_id'] >= 0) & df[date_col].notna()]
        
        if valid_data.empty:
            print("No valid data for aggregation")
            return pd.DataFrame()
        
        # Define aggregation functions
        agg_dict = {}
        for col in value_cols:
            if col in valid_data.columns:
                agg_dict[col] = ['mean', 'std', 'count']
        
        # Add metadata columns
        metadata_cols = ['State Name', 'County Name', 'Latitude_cluster', 'Longitude_cluster']
        for col in metadata_cols:
            if col in valid_data.columns:
                agg_dict[col] = 'first'
        
        # Perform aggregation
        try:
            aggregated = valid_data.groupby(['cluster_id', date_col]).agg(agg_dict).reset_index()
            
            # Flatten column names
            aggregated.columns = [
                '_'.join(col).strip('_') if col[1] else col[0] 
                for col in aggregated.columns
            ]
            
            print(f"Aggregated to {len(aggregated)} cluster-date combinations")
            return aggregated
            
        except Exception as e:
            print(f"Error during aggregation: {e}")
            return pd.DataFrame()
    
    def analyze_temporal_patterns(self, df: pd.DataFrame, 
                                value_col: str,
                                date_col: str = 'Date Local',
                                freq: str = 'D') -> Dict:
        """
        Analyze temporal patterns in environmental data.
        
        Args:
            df: DataFrame with time series data
            value_col: Column to analyze
            date_col: Date column name
            freq: Frequency for resampling ('D', 'W', 'M')
        
        Returns:
            Dictionary with analysis results
        """
        print(f"Analyzing temporal patterns for {value_col}...")
        
        try:
            # Prepare time series
            df_ts = df.copy()
            df_ts[date_col] = pd.to_datetime(df_ts[date_col])
            df_ts = df_ts.dropna(subset=[date_col, value_col])
            
            if df_ts.empty:
                return {'error': 'No valid data for temporal analysis'}
            
            # Create time series
            ts = df_ts.set_index(date_col)[value_col]
            ts_resampled = ts.resample(freq).mean()
            
            # Basic statistics
            results = {
                'basic_stats': {
                    'mean': float(ts_resampled.mean()),
                    'std': float(ts_resampled.std()),
                    'min': float(ts_resampled.min()),
                    'max': float(ts_resampled.max()),
                    'count': int(len(ts_resampled))
                },
                'temporal_range': {
                    'start_date': str(ts_resampled.index.min()),
                    'end_date': str(ts_resampled.index.max()),
                    'total_days': int((ts_resampled.index.max() - ts_resampled.index.min()).days)
                }
            }
            
            # Seasonal decomposition (if enough data)
            if len(ts_resampled) >= 365 and freq == 'D':
                try:
                    decomposition = seasonal_decompose(
                        ts_resampled.fillna(method='ffill'),
                        model='additive',
                        period=365
                    )
                    
                    results['seasonal_decomposition'] = {
                        'trend_mean': float(decomposition.trend.mean()),
                        'seasonal_amplitude': float(decomposition.seasonal.std()),
                        'residual_std': float(decomposition.resid.std())
                    }
                    
                except Exception as e:
                    results['seasonal_decomposition'] = {'error': str(e)}
            
            # Monthly patterns
            monthly_avg = ts_resampled.groupby(ts_resampled.index.month).mean()
            results['monthly_patterns'] = {
                'highest_month': int(monthly_avg.idxmax()),
                'lowest_month': int(monthly_avg.idxmin()),
                'seasonal_variation': float(monthly_avg.std())
            }
            
            # Year-over-year trends
            if ts_resampled.index.year.nunique() > 1:
                yearly_avg = ts_resampled.groupby(ts_resampled.index.year).mean()
                trend_slope = np.polyfit(range(len(yearly_avg)), yearly_avg.values, 1)[0]
                results['yearly_trend'] = {
                    'slope': float(trend_slope),
                    'direction': 'increasing' if trend_slope > 0 else 'decreasing'
                }
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def create_summary_report(self, epa_data: pd.DataFrame, 
                            weather_data: pd.DataFrame = None) -> Dict:
        """
        Create a comprehensive summary report of the environmental analysis.
        
        Args:
            epa_data: EPA air quality data
            weather_data: Weather data (optional)
        
        Returns:
            Dictionary containing summary statistics and insights
        """
        print("Generating summary report...")
        
        report = {
            'data_overview': {},
            'air_quality_summary': {},
            'temporal_analysis': {},
            'spatial_analysis': {},
            'data_quality': {}
        }
        
        # Data overview
        report['data_overview'] = {
            'epa_records': len(epa_data) if not epa_data.empty else 0,
            'weather_records': len(weather_data) if weather_data is not None and not weather_data.empty else 0,
            'date_range': {
                'start': str(epa_data['Date Local'].min()) if not epa_data.empty else 'N/A',
                'end': str(epa_data['Date Local'].max()) if not epa_data.empty else 'N/A'
            },
            'state': self.state_name,
            'processing_date': str(datetime.now().date())
        }
        
        # Air quality summary
        if not epa_data.empty:
            pollutant_summary = {}
            
            for pollutant in epa_data['Pollutant_Code'].unique():
                pollutant_data = epa_data[epa_data['Pollutant_Code'] == pollutant]
                
                if 'Arithmetic Mean' in pollutant_data.columns:
                    values = pollutant_data['Arithmetic Mean'].dropna()
                    if not values.empty:
                        pollutant_summary[pollutant] = {
                            'mean_concentration': float(values.mean()),
                            'max_concentration': float(values.max()),
                            'readings_count': int(len(values)),
                            'monitoring_sites': int(pollutant_data['Site Num'].nunique())
                        }
            
            report['air_quality_summary'] = pollutant_summary
            
            # Spatial analysis
            if 'Latitude' in epa_data.columns and 'Longitude' in epa_data.columns:
                coords = epa_data[['Latitude', 'Longitude']].dropna()
                if not coords.empty:
                    report['spatial_analysis'] = {
                        'monitoring_sites': int(len(coords)),
                        'geographic_extent': {
                            'lat_range': [float(coords['Latitude'].min()), float(coords['Latitude'].max())],
                            'lon_range': [float(coords['Longitude'].min()), float(coords['Longitude'].max())]
                        }
                    }
        
        # Data quality assessment
        if not epa_data.empty:
            quality_metrics = {}
            
            # Missing data analysis
            missing_data = epa_data.isnull().sum()
            quality_metrics['missing_data'] = {
                col: int(missing_data[col]) for col in missing_data.index 
                if missing_data[col] > 0
            }
            
            # Date completeness
            if 'Date Local' in epa_data.columns:
                epa_data['Date Local'] = pd.to_datetime(epa_data['Date Local'])
                date_range = pd.date_range(
                    start=epa_data['Date Local'].min(),
                    end=epa_data['Date Local'].max(),
                    freq='D'
                )
                actual_dates = len(epa_data['Date Local'].dt.date.unique())
                expected_dates = len(date_range)
                quality_metrics['temporal_completeness'] = {
                    'actual_days': actual_dates,
                    'expected_days': expected_dates,
                    'completeness_ratio': float(actual_dates / expected_dates) if expected_dates > 0 else 0
                }
            
            report['data_quality'] = quality_metrics
        
        return report
    
    def save_results(self, data: pd.DataFrame, filename: str, 
                    include_metadata: bool = True) -> str:
        """
        Save processed data with metadata.
        
        Args:
            data: DataFrame to save
            filename: Output filename
            include_metadata: Whether to include processing metadata
        
        Returns:
            Path to saved file
        """
        output_path = os.path.join(self.dirs['final_output'], filename)
        
        try:
            # Save main data
            if filename.endswith('.csv'):
                data.to_csv(output_path, index=False)
            elif filename.endswith('.parquet'):
                data.to_parquet(output_path, index=False)
            
            # Save metadata if requested
            if include_metadata:
                metadata = {
                    'processing_date': str(datetime.now()),
                    'state_code': self.state_code,
                    'state_name': self.state_name,
                    'data_shape': data.shape,
                    'columns': list(data.columns)
                }
                
                metadata_path = output_path.replace('.csv', '_metadata.json').replace('.parquet', '_metadata.json')
                
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            print(f"Saved results to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error saving results: {e}")
            return ""


def main():
    """Example usage of the Environmental Data Pipeline."""
    
    print("Environmental Air Quality Analysis Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = EnvironmentalDataPipeline(state_code=36)  # New York
    
    # Define pollutants to analyze
    pollutant_codes = [
        '88101',  # PM2.5
        '44201',  # Ozone
        '42401',  # SO2
        '42101',  # CO
        '42602'   # NO2
    ]
    
    print(f"Analyzing environmental data for {pipeline.state_name}")
    print(f"Pollutants: {pollutant_codes}")
    print(f"Time period: {pipeline.epa_start_year}-{pipeline.current_year-1}")
    
    # Note: Actual execution would require valid data sources
    print("\nPipeline initialized successfully!")
    print("To run full analysis, ensure you have:")
    print("1. Valid NOAA API token (set NOAA_API_TOKEN environment variable)")
    print("2. Internet connection for data downloads")
    print("3. Sufficient disk space (several GB for full dataset)")
    
    # Example workflow:
    print("\nExample workflow:")
    print("1. epa_data = pipeline.fetch_epa_pollutant_data(pollutant_codes)")
    print("2. weather_data = pipeline.fetch_weather_data()")
    print("3. clustered_data = pipeline.cluster_monitoring_stations(epa_data)")
    print("4. aggregated_data = pipeline.aggregate_by_cluster_and_date(clustered_data, ['Arithmetic Mean'])")
    print("5. report = pipeline.create_summary_report(epa_data, weather_data)")


if __name__ == "__main__":
    main()