import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def create_sample_crime_dataset(filename='sample_crime_data_update.csv', records=2000):
    """Create a sample crime dataset for demo purposes"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Indian cities and districts with coordinates
    locations = [
        {'district': 'Central Delhi', 'state': 'Delhi', 'lat': 28.6139, 'lon': 77.2090},
        {'district': 'South Delhi', 'state': 'Delhi', 'lat': 28.5355, 'lon': 77.2090},
        {'district': 'North Delhi', 'state': 'Delhi', 'lat': 28.7041, 'lon': 77.1025},
        {'district': 'East Delhi', 'state': 'Delhi', 'lat': 28.6508, 'lon': 77.3152},
        {'district': 'West Delhi', 'state': 'Delhi', 'lat': 28.6692, 'lon': 77.1174},
        {'district': 'Gurgaon', 'state': 'Haryana', 'lat': 28.4595, 'lon': 77.0266},
        {'district': 'Noida', 'state': 'Uttar Pradesh', 'lat': 28.5355, 'lon': 77.3910},
        {'district': 'Faridabad', 'state': 'Haryana', 'lat': 28.4089, 'lon': 77.3178},
        {'district': 'Mumbai Central', 'state': 'Maharashtra', 'lat': 19.0760, 'lon': 72.8777},
        {'district': 'Mumbai South', 'state': 'Maharashtra', 'lat': 18.9220, 'lon': 72.8347},
        {'district': 'Bangalore Urban', 'state': 'Karnataka', 'lat': 12.9716, 'lon': 77.5946},
        {'district': 'Chennai Central', 'state': 'Tamil Nadu', 'lat': 13.0827, 'lon': 80.2707},
        {'district': 'Kolkata Central', 'state': 'West Bengal', 'lat': 22.5726, 'lon': 88.3639},
        {'district': 'Jaipur', 'state': 'Rajasthan', 'lat': 26.9124, 'lon': 75.7873},
        {'district': 'Ahmedabad', 'state': 'Gujarat', 'lat': 23.0225, 'lon': 72.5714},
        {'district': 'Pune', 'state': 'Maharashtra', 'lat': 18.5204, 'lon': 73.8567},
        {'district': 'Hyderabad', 'state': 'Telangana', 'lat': 17.3850, 'lon': 78.4867},
        {'district': 'Lucknow', 'state': 'Uttar Pradesh', 'lat': 26.8467, 'lon': 80.9462}
    ]
    
    # Crime types with different probabilities
    crime_types = [
        'Theft', 'Burglary', 'Assault', 'Fraud', 'Vandalism', 
        'Drug Offense', 'Robbery', 'Domestic Violence', 'Cybercrime',
        'Vehicle Theft', 'Pickpocketing', 'Chain Snatching'
    ]
    
    # Crime type weights (some crimes are more common)
    crime_weights = [0.25, 0.15, 0.12, 0.10, 0.08, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02]
    
    # Generate sample data
    sample_data = []
    
    # Date range: last 18 months to current date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=540)  # 18 months
    
    for i in range(records):
        # Select random location
        location = random.choice(locations)
        
        # Generate random date
        random_days = random.randint(0, 540)
        crime_date = start_date + timedelta(days=random_days)
        
        # Add some time variation
        crime_hour = random.randint(0, 23)
        crime_minute = random.randint(0, 59)
        crime_datetime = crime_date.replace(hour=crime_hour, minute=crime_minute)
        
        # Select crime type based on weights
        crime_type = np.random.choice(crime_types, p=crime_weights)
        
        # Add some coordinate variation (within ~1km radius)
        lat_variation = np.random.normal(0, 0.005)  # ~500m variation
        lon_variation = np.random.normal(0, 0.005)
        
        # Create seasonal patterns (more crimes in certain months)
        month = crime_date.month
        seasonal_factor = 1.0
        if month in [4, 5, 6]:  # Summer months - higher crime
            seasonal_factor = 1.3
        elif month in [11, 12, 1]:  # Winter months - lower crime
            seasonal_factor = 0.8
        
        # Create time-of-day patterns
        hour_factor = 1.0
        if 18 <= crime_hour <= 23:  # Evening hours - higher crime
            hour_factor = 1.4
        elif 2 <= crime_hour <= 6:  # Early morning - lower crime
            hour_factor = 0.6
        
        # Create district-specific crime patterns
        district_factor = 1.0
        if 'Central' in location['district'] or 'Mumbai' in location['district']:
            district_factor = 1.2  # Urban centers have more crime
        elif location['district'] in ['Noida', 'Gurgaon']:
            district_factor = 0.9  # Planned cities have less crime
        
        # Determine if this record should exist based on factors
        probability = seasonal_factor * hour_factor * district_factor * 0.3
        if random.random() > probability:
            continue
        
        # Create the record
        record = {
            'date': crime_datetime.strftime('%Y-%m-%d'),
            'time': crime_datetime.strftime('%H:%M:%S'),
            'datetime': crime_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'district': location['district'],
            'state': location['state'],
            'crime_type': crime_type,
            'latitude': location['lat'] + lat_variation,
            'longitude': location['lon'] + lon_variation,
            'year': crime_date.year,
            'month': crime_date.month,
            'day': crime_date.day,
            'hour': crime_hour,
            'day_of_week': crime_date.weekday(),
            'is_weekend': 1 if crime_date.weekday() >= 5 else 0,
            'season': 'Summer' if month in [4,5,6] else 'Winter' if month in [11,12,1] else 'Monsoon' if month in [7,8,9] else 'Spring'
        }
        
        sample_data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Add some derived features
    df['crime_severity'] = df['crime_type'].map({
        'Theft': 'Medium', 'Burglary': 'High', 'Assault': 'High', 'Fraud': 'Medium',
        'Vandalism': 'Low', 'Drug Offense': 'High', 'Robbery': 'High', 
        'Domestic Violence': 'High', 'Cybercrime': 'Medium', 'Vehicle Theft': 'Medium',
        'Pickpocketing': 'Low', 'Chain Snatching': 'Medium'
    })
    
    # Add case ID
    df['case_id'] = ['CASE_' + str(i+1).zfill(6) for i in range(len(df))]
    
    # Sort by date
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Save to CSV
    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', filename)
    df.to_csv(filepath, index=False)
    
    print(f"✅ Sample dataset created: {filepath}")
    print(f"📊 Records: {len(df)}")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"🏙️ Districts: {df['district'].nunique()}")
    print(f"🚨 Crime types: {df['crime_type'].nunique()}")
    print(f"📈 Top districts by crime count:")
    print(df['district'].value_counts().head())
    print(f"📈 Top crime types:")
    print(df['crime_type'].value_counts().head())
    
    return filepath

def create_updated_dataset():
    """Create an updated dataset with recent data for demo"""
    
    # Create a dataset with more recent data and different patterns
    np.random.seed(123)  # Different seed for variation
    random.seed(123)
    
    # Focus on recent 6 months with higher crime rates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months
    
    # Updated locations with some new areas
    locations = [
        {'district': 'Central Delhi', 'state': 'Delhi', 'lat': 28.6139, 'lon': 77.2090},
        {'district': 'South Delhi', 'state': 'Delhi', 'lat': 28.5355, 'lon': 77.2090},
        {'district': 'North Delhi', 'state': 'Delhi', 'lat': 28.7041, 'lon': 77.1025},
        {'district': 'East Delhi', 'state': 'Delhi', 'lat': 28.6508, 'lon': 77.3152},
        {'district': 'West Delhi', 'state': 'Delhi', 'lat': 28.6692, 'lon': 77.1174},
        {'district': 'Gurgaon', 'state': 'Haryana', 'lat': 28.4595, 'lon': 77.0266},
        {'district': 'Noida', 'state': 'Uttar Pradesh', 'lat': 28.5355, 'lon': 77.3910},
        {'district': 'Greater Noida', 'state': 'Uttar Pradesh', 'lat': 28.4744, 'lon': 77.5040},  # New area
        {'district': 'Dwarka', 'state': 'Delhi', 'lat': 28.5921, 'lon': 77.0460},  # New area
        {'district': 'Rohini', 'state': 'Delhi', 'lat': 28.7041, 'lon': 77.1025}   # New area
    ]
    
    # Updated crime types with cybercrime increase
    crime_types = [
        'Cybercrime', 'Online Fraud', 'Theft', 'Burglary', 'Assault', 
        'Fraud', 'Vandalism', 'Drug Offense', 'Robbery', 'Vehicle Theft'
    ]
    
    # Updated weights (cybercrime increased)
    crime_weights = [0.20, 0.15, 0.15, 0.12, 0.10, 0.08, 0.06, 0.06, 0.04, 0.04]
    
    sample_data = []
    
    for i in range(1500):  # Smaller dataset for quick demo
        location = random.choice(locations)
        
        # Generate recent date
        random_days = random.randint(0, 180)
        crime_date = start_date + timedelta(days=random_days)
        
        crime_hour = random.randint(0, 23)
        crime_minute = random.randint(0, 59)
        crime_datetime = crime_date.replace(hour=crime_hour, minute=crime_minute)
        
        crime_type = np.random.choice(crime_types, p=crime_weights)
        
        lat_variation = np.random.normal(0, 0.003)
        lon_variation = np.random.normal(0, 0.003)
        
        record = {
            'date': crime_datetime.strftime('%Y-%m-%d'),
            'time': crime_datetime.strftime('%H:%M:%S'),
            'datetime': crime_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'district': location['district'],
            'state': location['state'],
            'crime_type': crime_type,
            'latitude': location['lat'] + lat_variation,
            'longitude': location['lon'] + lon_variation,
            'year': crime_date.year,
            'month': crime_date.month,
            'day': crime_date.day,
            'hour': crime_hour,
            'day_of_week': crime_date.weekday(),
            'is_weekend': 1 if crime_date.weekday() >= 5 else 0
        }
        
        sample_data.append(record)
    
    df = pd.DataFrame(sample_data)
    df['case_id'] = ['UPDATE_' + str(i+1).zfill(6) for i in range(len(df))]
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Save updated dataset
    filepath = os.path.join('uploads', 'updated_crime_data_2024.csv')
    df.to_csv(filepath, index=False)
    
    print(f"✅ Updated dataset created: {filepath}")
    print(f"📊 Records: {len(df)}")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"🆕 New districts: Greater Noida, Dwarka, Rohini")
    print(f"📈 Cybercrime incidents: {len(df[df['crime_type'].str.contains('Cyber|Online')])}")
    
    return filepath

if __name__ == "__main__":
    print("🔄 Creating sample datasets for demo...")
    
    # Create initial dataset
    initial_dataset = create_sample_crime_dataset()
    
    # Create updated dataset
    updated_dataset = create_updated_dataset()
    
    print("\n✅ Sample datasets created successfully!")
    print("📁 Files available for upload demo:")
    print(f"   1. {initial_dataset}")
    print(f"   2. {updated_dataset}")
    print("\n🎯 Use these files to demonstrate:")
    print("   • Dataset upload and validation")
    print("   • Automatic model retraining")
    print("   • Performance comparison")
    print("   • Real-time analytics updates")
