#!/usr/bin/env python3
"""
Test script to verify ML integration
"""

import os
import sys

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from ml.forecast_engine import ForecastEngine
    print("✓ ForecastEngine imported successfully")
    
    # Test forecast engine
    dataset_path = os.path.join(os.path.dirname(__file__), 'datasets', 'merged_crime_data.csv')
    models_path = os.path.join(os.path.dirname(__file__), 'models')
    
    if os.path.exists(dataset_path):
        print(f"✓ Dataset found: {dataset_path}")
        
        try:
            print("🔄 Initializing ForecastEngine...")
            forecast_engine = ForecastEngine(dataset_path, models_path)
            print("✓ ForecastEngine initialized successfully")
            
            print("🔄 Testing forecast for Central Delhi...")
            results = forecast_engine.forecast_district("Central Delhi", 6)
            
            print("✅ Real ML Forecast Results:")
            print(f"   Best Model: {results['best_model']}")
            print(f"   Models Available: {list(results['model_performance'].keys())}")
            print(f"   Forecast Data Points: {len(results['forecast_data'])}")
            
            # Show first forecast entry
            if results['forecast_data']:
                first_forecast = results['forecast_data'][0]
                print(f"   First Month Forecast: {first_forecast}")
                
            print("\n🎯 Model Performance:")
            for model, perf in results['model_performance'].items():
                print(f"   {model.upper()}: MAE = {perf['mae']:.2f}")
                
        except Exception as e:
            print(f"❌ Error testing forecast engine: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Dataset not found: {dataset_path}")
        
except ImportError as e:
    print(f"❌ Import error: {str(e)}")
except Exception as e:
    print(f"❌ Unexpected error: {str(e)}")
    import traceback
    traceback.print_exc()
