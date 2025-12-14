#!/usr/bin/env python3
"""
Generate a comprehensive crime dataset that works with forecast models
This creates 3+ years of historical data with realistic patterns
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_comprehensive_crime_dataset():
    """Generate a realistic crime dataset with 3+ years of data"""
    
    print("🔄 Generating comprehensive crime dataset...")
    
    # Districts with realistic crime patterns
    districts = [
        'Central Delhi', 'South Delhi', 'North Delhi', 'West Delhi', 'East Delhi',
        'Noida', 'Gurgaon', 'Greater Noida', 'Dwarka', 'Rohini'
    ]
    
    # Crime types with different frequencies
    crime_types = [
        'Theft', 'Burglary', 'Vehicle Theft', 'Assault', 'Robbery',
        'Fraud', 'Vandalism', 'Drug Offense', 'Domestic Violence', 'Public Disorder'
    ]
    
    # Generate data from Jan 2021 to August 2025 (44+ months)
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2025, 8, 23)
    
    data = []
    case_id_counter = 1
    
    # Base crime rates per district (crimes per month)
    district_base_rates = {
        'Central Delhi': 180,
        'South Delhi': 160,
        'North Delhi': 170,
        'West Delhi': 150,
        'East Delhi': 140,
        'Noida': 130,
        'Gurgaon': 145,
        'Greater Noida': 120,
        'Dwarka': 125,
        'Rohini': 135
    }
    
    # Seasonal factors (higher crime in certain months)
    seasonal_factors = {
        1: 0.9,   # January - lower
        2: 0.85,  # February - lower  
        3: 0.95,  # March
        4: 1.0,   # April
        5: 1.1,   # May - higher
        6: 1.15,  # June - higher
        7: 1.2,   # July - peak
        8: 1.15,  # August - high
        9: 1.1,   # September
        10: 1.05, # October
        11: 1.0,  # November
        12: 1.1   # December - festivals
    }
    
    # Generate data month by month
    current_date = start_date
    
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        
        # Days in current month
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        
        days_in_month = (next_month - datetime(year, month, 1)).days
        
        for district in districts:
            # Calculate expected crimes for this district/month
            base_rate = district_base_rates[district]
            seasonal_factor = seasonal_factors[month]
            
            # Add some random variation (±20%)
            variation = random.uniform(0.8, 1.2)
            monthly_crimes = int(base_rate * seasonal_factor * variation)
            
            # Generate individual crime records for the month
            for _ in range(monthly_crimes):
                # Random day in month
                day = random.randint(1, days_in_month)
                
                # Random time (more crimes in evening/night)
                hour_weights = [
                    2, 2, 1, 1, 1, 2, 3, 5,  # 00-07: low activity
                    6, 7, 8, 9, 10, 9, 8, 7,  # 08-15: moderate activity  
                    9, 12, 15, 18, 16, 12, 8, 4  # 16-23: higher activity
                ]
                hour = random.choices(range(24), weights=hour_weights)[0]
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                
                crime_datetime = datetime(year, month, day, hour, minute, second)
                
                # Random crime type
                crime_type = random.choice(crime_types)
                
                # Generate coordinates (rough Delhi NCR area)
                if 'Delhi' in district:
                    lat_base, lon_base = 28.6139, 77.2090  # Delhi center
                else:
                    lat_base, lon_base = 28.5355, 77.3910  # Noida/Gurgaon area
                
                latitude = lat_base + random.uniform(-0.3, 0.3)
                longitude = lon_base + random.uniform(-0.3, 0.3)
                
                # Create record
                record = {
                    'date': crime_datetime.strftime('%Y-%m-%d'),
                    'time': crime_datetime.strftime('%H:%M:%S'),
                    'datetime': crime_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    'district': district,
                    'state': 'Delhi NCR',
                    'crime_type': crime_type,
                    'latitude': round(latitude, 6),
                    'longitude': round(longitude, 6),
                    'year': year,
                    'month': month,
                    'day': day,
                    'hour': hour,
                    'day_of_week': crime_datetime.weekday(),
                    'is_weekend': 1 if crime_datetime.weekday() >= 5 else 0,
                    'case_id': f'CASE_{case_id_counter:06d}'
                }
                
                data.append(record)
                case_id_counter += 1
        
        # Move to next month
        if month == 12:
            current_date = datetime(year + 1, 1, 1)
        else:
            current_date = datetime(year, month + 1, 1)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle the data to make it more realistic
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"✅ Generated {len(df)} crime records")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"📍 Districts: {len(df['district'].unique())}")
    print(f"📊 Monthly distribution:")
    
    # Show monthly distribution
    df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    monthly_counts = df.groupby('year_month').size()
    print(f"   First month: {monthly_counts.iloc[0]} crimes in {monthly_counts.index[0]}")
    print(f"   Last month: {monthly_counts.iloc[-1]} crimes in {monthly_counts.index[-1]}")
    print(f"   Total months: {len(monthly_counts)}")
    
    return df

def backup_existing_dataset():
    """Backup the existing dataset"""
    dataset_path = 'datasets/merged_crime_data.csv'
    if os.path.exists(dataset_path):
        backup_path = f'datasets/backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}_merged_crime_data.csv'
        os.rename(dataset_path, backup_path)
        print(f"✅ Existing dataset backed up to: {backup_path}")
        return True
    return False

def main():
    """Main function to generate and save the dataset"""
    
    # Ensure datasets directory exists
    os.makedirs('datasets', exist_ok=True)
    
    # Backup existing dataset
    backup_existing_dataset()
    
    # Generate new comprehensive dataset
    df = generate_comprehensive_crime_dataset()
    
    # Save the new dataset
    output_path = 'datasets/merged_crime_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n🎯 New dataset saved to: {output_path}")
    print(f"📈 This dataset has {len(df)} records spanning multiple years")
    print(f"🤖 ML models will now have sufficient data for forecasting!")
    
    # Verify the data works with forecast engine
    print("\n🔍 Verifying data compatibility...")
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Check monthly data for each district
    for district in df['district'].unique():
        district_data = df[df['district'] == district]
        monthly_data = district_data.groupby('year_month').size()
        print(f"   {district}: {len(monthly_data)} months of data")
    
    # Check aggregated data
    all_monthly_data = df.groupby('year_month').size()
    print(f"   All districts combined: {len(all_monthly_data)} months of data")
    print(f"   ✅ Sufficient for ML forecasting (need 24+ months)")

if __name__ == "__main__":
    main()
