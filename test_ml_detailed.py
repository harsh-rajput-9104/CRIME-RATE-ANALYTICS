#!/usr/bin/env python3
"""
Test script to check available districts and data
"""

import os
import sys
import pandas as pd

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from ml.forecast_engine import ForecastEngine
    print("✓ ForecastEngine imported successfully")
    
    # Check dataset first
    dataset_path = os.path.join(os.path.dirname(__file__), 'datasets', 'merged_crime_data.csv')
    
    if os.path.exists(dataset_path):
        print(f"✓ Dataset found: {dataset_path}")
        
        # Load and analyze the data
        df = pd.read_csv(dataset_path)
        print(f"📊 Dataset info: {len(df)} records")
        
        if 'district' in df.columns:
            districts = df['district'].unique()
            print(f"📍 Available districts: {list(districts)}")
            
            # Check data volume per district
            district_counts = df['district'].value_counts()
            print("\n📈 Records per district:")
            for district, count in district_counts.head(10).items():
                print(f"   {district}: {count} records")
        
        # Test with 'all' districts
        try:
            print("\n🔄 Initializing ForecastEngine...")
            forecast_engine = ForecastEngine(dataset_path, 
                                           os.path.join(os.path.dirname(__file__), 'models'))
            print("✓ ForecastEngine initialized successfully")
            
            print("🔄 Testing forecast for 'all' districts...")
            results = forecast_engine.forecast_district("all", 6)
            
            print("✅ Real ML Forecast Results:")
            print(f"   Best Model: {results['best_model']}")
            print(f"   Models Available: {list(results['model_performance'].keys())}")
            print(f"   Forecast Data Points: {len(results['forecast_data'])}")
            
            print("\n🎯 Model Performance:")
            for model, perf in results['model_performance'].items():
                print(f"   {model.upper()}: MAE = {perf['mae']:.2f}")
                
            # Show first few forecast points
            print("\n📅 First 3 forecast points:")
            for i, forecast in enumerate(results['forecast_data'][:3]):
                print(f"   {forecast['month']}: ARIMA={forecast.get('arima', 'N/A')}, "
                      f"SARIMA={forecast.get('sarima', 'N/A')}, LSTM={forecast.get('lstm', 'N/A')}")
                
        except Exception as e:
            print(f"❌ Error testing forecast engine: {str(e)}")
            import traceback
            traceback.print_exc()
            
    else:
        print(f"❌ Dataset not found: {dataset_path}")
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
