#!/usr/bin/env python3
"""
Check dataset structure
"""

import pandas as pd
import os

dataset_path = os.path.join(os.path.dirname(__file__), 'datasets', 'merged_crime_data.csv')

if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
    print("📊 Dataset Columns:")
    print(df.columns.tolist())
    print("\n📋 First 5 rows:")
    print(df.head())
    print("\n📈 Data types:")
    print(df.dtypes)
    
    # Check for date-related columns
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'month' in col.lower() or 'year' in col.lower()]
    print(f"\n📅 Date-related columns: {date_columns}")
    
    if date_columns:
        for col in date_columns:
            print(f"\n{col} sample values:")
            print(df[col].head(10).tolist())
else:
    print("❌ Dataset not found")
