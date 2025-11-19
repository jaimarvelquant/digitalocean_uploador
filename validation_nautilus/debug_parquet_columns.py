#!/usr/bin/env python3
"""
Debug script to examine Parquet data structure and fix column mapping issues
"""

import pandas as pd
from nautilus_validation.config import ConfigManager
from nautilus_validation.connectors import SpacesConnector

def debug_parquet_data():
    """Debug Parquet data structure to understand available columns"""
    print("=" * 60)
    print("DEBUGGING PARQUET DATA STRUCTURE")
    print("=" * 60)

    try:
        # Load config and connect to Spaces
        config = ConfigManager('config.yaml')
        do_config = config.get_digital_ocean_config()
        do_config['bucket_name'] = 'historical-db-tick'  # Use a known bucket
        spaces = SpacesConnector(do_config)

        # Find some AARTIIND PUT parquet files (based on the error above)
        prefix = 'nautilas-data/'
        all_objects = spaces.client.list_objects_v2(
            Bucket='historical-db-tick',
            Prefix=prefix,
            MaxKeys=1000
        )

        files = [obj['Key'] for obj in all_objects.get('Contents', [])
                if '.parquet' in obj['Key'].lower() and 'aartiind' in obj['Key'].lower()]

        print(f"Found {len(files)} AARTIIND parquet files:")
        for i, file in enumerate(files[:5]):  # Show first 5
            print(f"  {i+1}. {file}")

        if files:
            # Read a sample file
            sample_file = files[0]
            print(f"\nReading sample file: {sample_file}")

            sample_df = spaces.read_parquet_file(sample_file)
            print(f"Shape: {sample_df.shape}")
            print(f"Columns: {list(sample_df.columns)}")

            # Show data types
            print(f"\nData types:")
            for col in sample_df.columns:
                print(f"  {col}: {sample_df[col].dtype}")

            # Show sample data
            print(f"\nSample data (first 3 rows):")
            print(sample_df.head(3).to_string())

            # Check for date/time related columns
            date_time_cols = [col for col in sample_df.columns
                             if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp'])]
            print(f"\nDate/Time related columns: {date_time_cols}")

            # Check for OHLC columns
            ohlc_cols = [col for col in sample_df.columns
                        if any(keyword in col.upper() for keyword in ['OPEN', 'HIGH', 'LOW', 'CLOSE'])]
            print(f"OHLC columns: {ohlc_cols}")

            # Check for symbol column
            symbol_cols = [col for col in sample_df.columns if 'symbol' in col.lower()]
            print(f"Symbol columns: {symbol_cols}")

            # Check if there are any identifier columns that could be used for matching
            id_cols = []
            for col in sample_df.columns:
                unique_count = sample_df[col].nunique()
                total_count = len(sample_df)
                if unique_count == total_count:  # All values unique
                    id_cols.append((col, unique_count))
            print(f"\nPotential identifier columns (all unique): {id_cols}")

        else:
            print("No AARTIIND parquet files found")
            # Let's just find any parquet files
            all_parquet_files = [obj['Key'] for obj in all_objects.get('Contents', [])
                               if '.parquet' in obj['Key'].lower()]
            if all_parquet_files:
                print(f"\nFound {len(all_parquet_files)} total parquet files, showing first 5:")
                for i, file in enumerate(all_parquet_files[:5]):
                    print(f"  {i+1}. {file}")
            else:
                print("No parquet files found at all")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    debug_parquet_data()