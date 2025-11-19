#!/usr/bin/env python3
"""
Fix script to handle timestamp columns and data quality issues
"""

import pandas as pd
import numpy as np
from datetime import datetime

def extract_datetime_from_parquet(df):
    """
    Enhanced datetime extraction that handles various timestamp formats
    """
    print("Extracting datetime information from Parquet data...")
    df_with_datetime = df.copy()

    # Check if we already have separate date and time columns
    if 'date' in df.columns and 'time' in df.columns:
        print("  Found existing date and time columns")
        return df_with_datetime

    # Look for timestamp columns (this is what your data has!)
    timestamp_cols = [col for col in df.columns
                     if any(keyword in col.lower() for keyword in ['ts_event', 'ts_init', 'timestamp', 'datetime', 'date_time'])]

    if timestamp_cols:
        # Prefer ts_event over ts_init for the primary timestamp
        timestamp_col = 'ts_event' if 'ts_event' in timestamp_cols else timestamp_cols[0]
        print(f"  Found timestamp column: {timestamp_col}")

        try:
            # Handle different timestamp formats
            timestamp_values = df_with_datetime[timestamp_col]

            # Check if it's Unix timestamp (uint64)
            if timestamp_values.dtype == 'uint64' or timestamp_values.dtype == 'int64':
                print(f"  Detected Unix timestamp format")
                # Convert Unix timestamp to datetime
                df_with_datetime['datetime'] = pd.to_datetime(timestamp_values, unit='ns')
            else:
                # Try to convert directly to datetime
                df_with_datetime['datetime'] = pd.to_datetime(timestamp_values)

            # Extract date and time components in MySQL format
            df_with_datetime['date'] = df_with_datetime['datetime'].dt.strftime('%y%m%d').astype(int)
            df_with_datetime['time'] = df_with_datetime['datetime'].dt.hour * 10000 + \
                                     df_with_datetime['datetime'].dt.minute * 100 + \
                                     df_with_datetime['datetime'].dt.second

            print(f"  Successfully extracted date and time from {timestamp_col}")
            print(f"  Sample converted datetime: {df_with_datetime[['date', 'time', 'datetime']].iloc[0].tolist()}")

            return df_with_datetime

        except Exception as e:
            print(f"  Error extracting datetime: {e}")
            print(f"  Timestamp dtype: {timestamp_values.dtype}")
            print(f"  Sample timestamp values: {timestamp_values.head().tolist()}")

    # If no timestamp columns found, show available columns
    print(f"  No suitable timestamp columns found")
    print(f"  Available columns: {list(df.columns)}")
    return df_with_datetime

def debug_parquet_data_structure(df):
    """Debug the parquet data structure to understand the format"""
    print("=" * 60)
    print("DEBUGGING PARQUET DATA STRUCTURE")
    print("=" * 60)

    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")

    print(f"\nFirst few rows:")
    print(df.head())

    print(f"\nTimestamp column analysis:")
    timestamp_cols = [col for col in df.columns if 'ts' in col.lower() or 'time' in col.lower()]
    for col in timestamp_cols:
        print(f"\n{col}:")
        print(f"  Dtype: {df[col].dtype}")
        print(f"  Sample values: {df[col].head().tolist()}")
        if df[col].dtype in ['uint64', 'int64']:
            print(f"  Min: {df[col].min()}, Max: {df[col].max()}")
            # Try to convert a few timestamps to dates for inspection
            try:
                sample_dates = pd.to_datetime(df[col].head(), unit='ns')
                print(f"  Converted dates: {sample_dates.tolist()}")
            except:
                print(f"  Could not convert to dates")

def test_timestamp_extraction():
    """Test timestamp extraction with sample data"""
    print("=" * 60)
    print("TESTING TIMESTAMP EXTRACTION")
    print("=" * 60)

    # Create sample data similar to your structure
    current_time = datetime.now()
    base_timestamp = int(current_time.timestamp() * 1e9)  # Convert to nanoseconds

    sample_data = {
        'open': [100.0, 101.0, 102.0],
        'high': [105.0, 106.0, 107.0],
        'low': [95.0, 96.0, 97.0],
        'close': [104.0, 105.0, 106.0],
        'volume': [1000, 1200, 800],
        'ts_event': [base_timestamp + i * 60000000000 for i in range(3)],  # Add 1 minute intervals
        'ts_init': [base_timestamp + i * 60000000000 for i in range(3)]
    }

    sample_df = pd.DataFrame(sample_data)
    print("Created sample data:")
    print(f"  Shape: {sample_df.shape}")
    print(f"  Columns: {list(sample_df.columns)}")
    print(f"  ts_event sample: {sample_df['ts_event'].tolist()}")

    # Test extraction
    result_df = extract_datetime_from_parquet(sample_df)

    if 'date' in result_df.columns and 'time' in result_df.columns:
        print("✅ Timestamp extraction successful!")
        print(f"  Extracted date/time: {result_df[['date', 'time']].head()}")
    else:
        print("❌ Timestamp extraction failed!")

if __name__ == "__main__":
    test_timestamp_extraction()