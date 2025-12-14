#!/usr/bin/env python3
"""
Model Retraining Script for Crime Rate Analytics
Retrains forecasting and risk classification models on updated merged dataset
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Add backend path to import ML modules
sys.path.append('../backend')

from ml.forecast_engine import ForecastEngine
from ml.risk_classifier import RiskClassifier

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelRetrainer:
    def __init__(self, dataset_path, models_path='../backend/models/'):
        self.dataset_path = dataset_path
        self.models_path = models_path
        self.data = None
        
        # Ensure models directory exists
        os.makedirs(models_path, exist_ok=True)
        
    def load_data(self):
        """Load the crime dataset"""
        try:
            logger.info(f"Loading data from {self.dataset_path}")
            self.data = pd.read_csv(self.dataset_path)
            logger.info(f"Data loaded successfully: {len(self.data)} records")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def validate_data(self):
        """Validate data for model training"""
        try:
            logger.info("Validating data for model training...")
            
            required_columns = ['date', 'district']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check date format
            try:
                self.data['date'] = pd.to_datetime(self.data['date'])
            except Exception as e:
                logger.error(f"Error parsing dates: {str(e)}")
                return False
            
            # Check for sufficient data
            if len(self.data) < 100:
                logger.error("Insufficient data for model training (minimum 100 records required)")
                return False
            
            # Check date range
            date_range = (self.data['date'].max() - self.data['date'].min()).days
            if date_range < 365:
                logger.warning("Less than 1 year of data available. Models may not perform optimally.")
            
            logger.info("Data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return False
    
    def retrain_forecast_models(self):
        """Retrain all forecasting models"""
        try:
            logger.info("Starting forecast model retraining...")
            
            # Initialize forecast engine
            forecast_engine = ForecastEngine(self.dataset_path, self.models_path)
            
            # Get unique districts
            districts = self.data['district'].unique()
            logger.info(f"Training models for {len(districts)} districts")
            
            # Train models for each district
            training_results = {}
            
            for district in districts:
                try:
                    logger.info(f"Training models for district: {district}")
                    
                    # Prepare district data
                    district_data = forecast_engine.prepare_district_data(district)
                    
                    if len(district_data) < 24:  # Need at least 2 years of monthly data
                        logger.warning(f"Insufficient data for district {district} (need 24+ months)")
                        continue
                    
                    # Split data for training and validation
                    train_size = int(len(district_data) * 0.8)
                    train_data = district_data[:train_size]
                    test_data = district_data[train_size:]
                    
                    district_results = {}
                    
                    # Train ARIMA model
                    try:
                        arima_model = forecast_engine.train_arima_model(train_data)
                        if arima_model:
                            # Evaluate model
                            if len(test_data) > 0:
                                test_forecast = arima_model.forecast(steps=len(test_data))
                                mae = np.mean(np.abs(test_data['crime_count'] - test_forecast))
                                district_results['arima'] = {'mae': mae, 'model': arima_model}
                                logger.info(f"ARIMA model trained for {district} - MAE: {mae:.2f}")
                    except Exception as e:
                        logger.warning(f"ARIMA training failed for {district}: {str(e)}")
                    
                    # Train SARIMA model
                    try:
                        sarima_model = forecast_engine.train_sarima_model(train_data)
                        if sarima_model:
                            # Evaluate model
                            if len(test_data) > 0:
                                test_forecast = sarima_model.forecast(steps=len(test_data))
                                mae = np.mean(np.abs(test_data['crime_count'] - test_forecast))
                                district_results['sarima'] = {'mae': mae, 'model': sarima_model}
                                logger.info(f"SARIMA model trained for {district} - MAE: {mae:.2f}")
                    except Exception as e:
                        logger.warning(f"SARIMA training failed for {district}: {str(e)}")
                    
                    # Train LSTM model
                    try:
                        X, y, scaler = forecast_engine.prepare_lstm_data(train_data)
                        if X is not None and y is not None:
                            lstm_model = forecast_engine.train_lstm_model(X, y)
                            if lstm_model:
                                # Evaluate model
                                if len(test_data) > 0 and len(X) > 0:
                                    # Simple evaluation for LSTM
                                    test_predictions = []
                                    test_sequence = X[-1].reshape(1, -1, 1)
                                    
                                    for _ in range(len(test_data)):
                                        pred = lstm_model.predict(test_sequence, verbose=0)
                                        test_predictions.append(pred[0, 0])
                                        test_sequence = np.roll(test_sequence, -1, axis=1)
                                        test_sequence[0, -1, 0] = pred[0, 0]
                                    
                                    test_predictions = scaler.inverse_transform(
                                        np.array(test_predictions).reshape(-1, 1)
                                    ).flatten()
                                    
                                    mae = np.mean(np.abs(test_data['crime_count'] - test_predictions))
                                    district_results['lstm'] = {'mae': mae, 'model': lstm_model, 'scaler': scaler}
                                    logger.info(f"LSTM model trained for {district} - MAE: {mae:.2f}")
                    except Exception as e:
                        logger.warning(f"LSTM training failed for {district}: {str(e)}")
                    
                    training_results[district] = district_results
                    
                except Exception as e:
                    logger.error(f"Error training models for district {district}: {str(e)}")
                    continue
            
            # Save training results summary
            self.save_training_summary('forecast', training_results)
            
            logger.info("Forecast model retraining completed")
            return True
            
        except Exception as e:
            logger.error(f"Error retraining forecast models: {str(e)}")
            return False
    
    def retrain_risk_classifier(self):
        """Retrain the risk classification model"""
        try:
            logger.info("Starting risk classifier retraining...")
            
            # Initialize risk classifier
            risk_classifier = RiskClassifier(self.dataset_path, self.models_path)
            
            # The RiskClassifier automatically trains during initialization
            # Get training results
            training_results = {
                'model_accuracy': risk_classifier.model_accuracy,
                'feature_names': risk_classifier.feature_names,
                'num_features': len(risk_classifier.feature_names) if risk_classifier.feature_names else 0,
                'num_districts': len(risk_classifier.feature_data) if hasattr(risk_classifier, 'feature_data') else 0
            }
            
            # Save training results summary
            self.save_training_summary('risk_classifier', training_results)
            
            logger.info(f"Risk classifier retrained - Accuracy: {risk_classifier.model_accuracy:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Error retraining risk classifier: {str(e)}")
            return False
    
    def save_training_summary(self, model_type, results):
        """Save training summary to file"""
        try:
            import json
            
            summary = {
                'model_type': model_type,
                'training_date': datetime.now().isoformat(),
                'dataset_path': self.dataset_path,
                'results': results
            }
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Clean results for JSON serialization
            clean_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    clean_value = {}
                    for k, v in value.items():
                        if k not in ['model', 'scaler']:  # Skip model objects
                            clean_value[k] = convert_numpy(v)
                    clean_results[key] = clean_value
                else:
                    clean_results[key] = convert_numpy(value)
            
            summary['results'] = clean_results
            
            summary_file = os.path.join(self.models_path, f'{model_type}_training_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Training summary saved: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving training summary: {str(e)}")
    
    def retrain_all_models(self):
        """Retrain all models"""
        logger.info("Starting complete model retraining...")
        
        # Load and validate data
        if not self.load_data():
            return False
        
        if not self.validate_data():
            return False
        
        # Retrain forecast models
        forecast_success = self.retrain_forecast_models()
        
        # Retrain risk classifier
        risk_success = self.retrain_risk_classifier()
        
        if forecast_success and risk_success:
            logger.info("All models retrained successfully!")
            return True
        else:
            logger.warning("Some models failed to retrain")
            return False

def main():
    parser = argparse.ArgumentParser(description='Retrain crime analytics models')
    parser.add_argument('--dataset', '-d', required=True, help='Path to crime dataset CSV file')
    parser.add_argument('--models-dir', '-m', default='../backend/models/', help='Models directory')
    parser.add_argument('--model-type', '-t', choices=['forecast', 'risk', 'all'], default='all',
                       help='Type of models to retrain')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    # Create retrainer
    retrainer = ModelRetrainer(args.dataset, args.models_dir)
    
    # Retrain models based on type
    success = False
    if args.model_type == 'forecast':
        success = retrainer.load_data() and retrainer.validate_data() and retrainer.retrain_forecast_models()
    elif args.model_type == 'risk':
        success = retrainer.load_data() and retrainer.validate_data() and retrainer.retrain_risk_classifier()
    else:  # all
        success = retrainer.retrain_all_models()
    
    if success:
        logger.info("Model retraining completed successfully!")
        sys.exit(0)
    else:
        logger.error("Model retraining failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
