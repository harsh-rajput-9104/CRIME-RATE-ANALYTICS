#!/usr/bin/env python3
"""
Data Preprocessing Script for Crime Rate Analytics
Preprocesses merged CSV for ML models (monthly/district grouping, missing value handling)
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CrimeDataPreprocessor:
    def __init__(self, input_file, output_dir='../backend/datasets/'):
        self.input_file = input_file
        self.output_dir = output_dir
        self.data = None
        self.processed_data = None
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load the merged crime data"""
        try:
            logger.info(f"Loading data from {self.input_file}")
            self.data = pd.read_csv(self.input_file)
            logger.info(f"Data loaded successfully: {len(self.data)} records")
            logger.info(f"Columns: {list(self.data.columns)}")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def clean_data(self):
        """Clean and standardize the data"""
        try:
            logger.info("Starting data cleaning...")
            
            # Make a copy for processing
            self.processed_data = self.data.copy()
            
            # Standardize column names
            column_mapping = {
                'Date': 'date',
                'DATE': 'date',
                'District': 'district',
                'DISTRICT': 'district',
                'Crime_Type': 'crime_type',
                'CRIME_TYPE': 'crime_type',
                'Crime Type': 'crime_type',
                'Latitude': 'latitude',
                'LATITUDE': 'latitude',
                'LAT': 'latitude',
                'Longitude': 'longitude',
                'LONGITUDE': 'longitude',
                'LON': 'longitude',
                'LNG': 'longitude'
            }
            
            # Rename columns if they exist
            for old_name, new_name in column_mapping.items():
                if old_name in self.processed_data.columns:
                    self.processed_data.rename(columns={old_name: new_name}, inplace=True)
            
            # Convert date column to datetime
            if 'date' in self.processed_data.columns:
                self.processed_data['date'] = pd.to_datetime(self.processed_data['date'], errors='coerce')
                logger.info(f"Date range: {self.processed_data['date'].min()} to {self.processed_data['date'].max()}")
            else:
                logger.warning("No date column found")
            
            # Clean district names
            if 'district' in self.processed_data.columns:
                self.processed_data['district'] = self.processed_data['district'].astype(str).str.strip().str.title()
                logger.info(f"Unique districts: {self.processed_data['district'].nunique()}")
            
            # Clean crime types
            if 'crime_type' in self.processed_data.columns:
                self.processed_data['crime_type'] = self.processed_data['crime_type'].astype(str).str.strip().str.title()
                logger.info(f"Unique crime types: {self.processed_data['crime_type'].nunique()}")
            
            # Handle missing values
            self.handle_missing_values()
            
            # Remove duplicates
            initial_count = len(self.processed_data)
            self.processed_data.drop_duplicates(inplace=True)
            final_count = len(self.processed_data)
            logger.info(f"Removed {initial_count - final_count} duplicate records")
            
            logger.info("Data cleaning completed")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return False
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        try:
            logger.info("Handling missing values...")
            
            # Check for missing values
            missing_summary = self.processed_data.isnull().sum()
            logger.info(f"Missing values summary:\n{missing_summary[missing_summary > 0]}")
            
            # Handle missing dates
            if 'date' in self.processed_data.columns:
                date_missing = self.processed_data['date'].isnull().sum()
                if date_missing > 0:
                    logger.warning(f"Removing {date_missing} records with missing dates")
                    self.processed_data = self.processed_data.dropna(subset=['date'])
            
            # Handle missing districts
            if 'district' in self.processed_data.columns:
                district_missing = self.processed_data['district'].isnull().sum()
                if district_missing > 0:
                    logger.warning(f"Filling {district_missing} missing districts with 'Unknown'")
                    self.processed_data['district'].fillna('Unknown', inplace=True)
            
            # Handle missing crime types
            if 'crime_type' in self.processed_data.columns:
                crime_type_missing = self.processed_data['crime_type'].isnull().sum()
                if crime_type_missing > 0:
                    logger.warning(f"Filling {crime_type_missing} missing crime types with 'Other'")
                    self.processed_data['crime_type'].fillna('Other', inplace=True)
            
            # Handle missing coordinates
            if 'latitude' in self.processed_data.columns and 'longitude' in self.processed_data.columns:
                coord_missing = self.processed_data[['latitude', 'longitude']].isnull().any(axis=1).sum()
                if coord_missing > 0:
                    logger.info(f"Found {coord_missing} records with missing coordinates (will use synthetic coordinates)")
            
            logger.info("Missing value handling completed")
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
    
    def create_time_features(self):
        """Create time-based features for analysis"""
        try:
            logger.info("Creating time-based features...")
            
            if 'date' not in self.processed_data.columns:
                logger.warning("No date column found, skipping time feature creation")
                return False
            
            # Extract time components
            self.processed_data['year'] = self.processed_data['date'].dt.year
            self.processed_data['month'] = self.processed_data['date'].dt.month
            self.processed_data['day'] = self.processed_data['date'].dt.day
            self.processed_data['hour'] = self.processed_data['date'].dt.hour
            self.processed_data['day_of_week'] = self.processed_data['date'].dt.dayofweek
            self.processed_data['day_name'] = self.processed_data['date'].dt.day_name()
            self.processed_data['month_name'] = self.processed_data['date'].dt.month_name()
            
            # Create period features
            self.processed_data['year_month'] = self.processed_data['date'].dt.to_period('M')
            self.processed_data['quarter'] = self.processed_data['date'].dt.quarter
            
            # Create categorical time features
            self.processed_data['is_weekend'] = self.processed_data['day_of_week'].isin([5, 6])
            self.processed_data['time_of_day'] = pd.cut(
                self.processed_data['hour'], 
                bins=[0, 6, 12, 18, 24], 
                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                include_lowest=True
            )
            
            logger.info("Time-based features created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating time features: {str(e)}")
            return False
    
    def create_aggregated_datasets(self):
        """Create aggregated datasets for different analysis purposes"""
        try:
            logger.info("Creating aggregated datasets...")
            
            aggregated_datasets = {}
            
            # Monthly district aggregation
            if 'year_month' in self.processed_data.columns and 'district' in self.processed_data.columns:
                monthly_district = self.processed_data.groupby(['year_month', 'district']).agg({
                    'crime_type': ['count', 'nunique'],
                    'is_weekend': 'sum',
                    'hour': ['mean', 'std']
                }).reset_index()
                
                # Flatten column names
                monthly_district.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                          for col in monthly_district.columns.values]
                
                # Rename columns for clarity
                monthly_district.rename(columns={
                    'crime_type_count': 'total_crimes',
                    'crime_type_nunique': 'unique_crime_types',
                    'is_weekend_sum': 'weekend_crimes',
                    'hour_mean': 'avg_hour',
                    'hour_std': 'hour_std'
                }, inplace=True)
                
                aggregated_datasets['monthly_district'] = monthly_district
                logger.info(f"Monthly district aggregation: {len(monthly_district)} records")
            
            # Daily aggregation
            if 'date' in self.processed_data.columns:
                daily_agg = self.processed_data.groupby(self.processed_data['date'].dt.date).agg({
                    'crime_type': 'count',
                    'district': 'nunique'
                }).reset_index()
                
                daily_agg.rename(columns={
                    'crime_type': 'daily_crimes',
                    'district': 'districts_affected'
                }, inplace=True)
                
                aggregated_datasets['daily'] = daily_agg
                logger.info(f"Daily aggregation: {len(daily_agg)} records")
            
            # Crime type analysis
            if 'crime_type' in self.processed_data.columns and 'district' in self.processed_data.columns:
                crime_type_district = self.processed_data.groupby(['district', 'crime_type']).size().reset_index(name='count')
                aggregated_datasets['crime_type_district'] = crime_type_district
                logger.info(f"Crime type by district: {len(crime_type_district)} records")
            
            return aggregated_datasets
            
        except Exception as e:
            logger.error(f"Error creating aggregated datasets: {str(e)}")
            return {}
    
    def save_processed_data(self, aggregated_datasets):
        """Save all processed datasets"""
        try:
            logger.info("Saving processed datasets...")
            
            # Save main processed dataset
            main_output_file = os.path.join(self.output_dir, 'processed_crime_data.csv')
            self.processed_data.to_csv(main_output_file, index=False)
            logger.info(f"Main processed dataset saved: {main_output_file}")
            
            # Save aggregated datasets
            for name, dataset in aggregated_datasets.items():
                output_file = os.path.join(self.output_dir, f'{name}_aggregated.csv')
                dataset.to_csv(output_file, index=False)
                logger.info(f"Aggregated dataset saved: {output_file}")
            
            # Create data summary
            self.create_data_summary()
            
            logger.info("All datasets saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            return False
    
    def create_data_summary(self):
        """Create a summary report of the processed data"""
        try:
            summary = {
                'processing_date': datetime.now().isoformat(),
                'total_records': len(self.processed_data),
                'date_range': {
                    'start': str(self.processed_data['date'].min()) if 'date' in self.processed_data.columns else None,
                    'end': str(self.processed_data['date'].max()) if 'date' in self.processed_data.columns else None
                },
                'unique_districts': self.processed_data['district'].nunique() if 'district' in self.processed_data.columns else 0,
                'unique_crime_types': self.processed_data['crime_type'].nunique() if 'crime_type' in self.processed_data.columns else 0,
                'columns': list(self.processed_data.columns),
                'missing_values': self.processed_data.isnull().sum().to_dict()
            }
            
            summary_file = os.path.join(self.output_dir, 'data_summary.json')
            import json
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Data summary saved: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error creating data summary: {str(e)}")
    
    def process(self):
        """Main processing pipeline"""
        logger.info("Starting data preprocessing pipeline...")
        
        # Load data
        if not self.load_data():
            return False
        
        # Clean data
        if not self.clean_data():
            return False
        
        # Create time features
        self.create_time_features()
        
        # Create aggregated datasets
        aggregated_datasets = self.create_aggregated_datasets()
        
        # Save processed data
        if not self.save_processed_data(aggregated_datasets):
            return False
        
        logger.info("Data preprocessing completed successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Preprocess crime data for ML models')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', default='../backend/datasets/', help='Output directory')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Create preprocessor and run
    preprocessor = CrimeDataPreprocessor(args.input, args.output)
    success = preprocessor.process()
    
    if success:
        logger.info("Preprocessing completed successfully!")
        sys.exit(0)
    else:
        logger.error("Preprocessing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
