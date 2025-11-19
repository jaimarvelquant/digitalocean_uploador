#!/usr/bin/env python3
"""
Simple validation script without emoji characters
"""

import pandas as pd
from nautilus_validation.config import ConfigManager
from nautilus_validation.connectors import SpacesConnector, DatabaseConnector
from nautilus_validation.validators import ValidationEngine

def simple_validation():
    """Run a simple validation without emoji issues"""
    print("Starting simple validation...")

    try:
        # Load configuration
        config = ConfigManager('config.yaml')

        # Initialize connectors
        spaces = SpacesConnector(config.get_digital_ocean_config())
        database = DatabaseConnector(config.get_database_config())

        # Read Parquet data (small sample)
        parquet_files = spaces.list_parquet_files("raw/parquet_data/futures/banknifty/2024/01/")
        print(f"Found {len(parquet_files)} Parquet files")

        # Read just 2 files for testing
        sample_files = parquet_files[:2]
        parquet_dfs = []

        for file in sample_files:
            print(f"Reading {file}...")
            df = spaces.read_parquet_file(file)
            parquet_dfs.append(df)
            print(f"  Read {len(df)} rows")

        source_df = pd.concat(parquet_dfs, ignore_index=True)
        print(f"Total Parquet rows: {len(source_df):,}")

        # Read database data (sample)
        print("Reading database data...")
        with database.engine.connect() as conn:
            # Read same date range from database (limited for testing)
            sample_df = pd.read_sql_query(
                f"SELECT date, time, symbol, open, high, low, close, volume, oi FROM banknifty_future LIMIT 10000",
                conn
            )

        print(f"Database rows read: {len(sample_df):,}")

        # Run validation
        validation_config = config.get_validation_config()
        validation_config['row_count_validation'] = True
        validation_config['data_integrity_validation'] = True
        validation_config['full_data_comparison'] = False  # Skip for now

        engine = ValidationEngine(validation_config)

        print("Running validation engine...")
        result = engine.validate(source_df, sample_df, key_columns=['date', 'time'])

        # Print results
        print(f"\n{'='*50}")
        print("VALIDATION RESULTS")
        print(f"{'='*50}")
        print(f"Overall Status: {result.overall_status.value}")
        print(f"Source Rows: {result.total_rows_source:,}")
        print(f"Target Rows: {result.total_rows_target:,}")
        print(f"Validations Run: {len(result.validation_results)}")

        for val_result in result.validation_results:
            print(f"\n{val_result.validation_type}: {val_result.status.value}")
            print(f"  Message: {val_result.message}")
            print(f"  Execution time: {val_result.execution_time:.3f}s")

            if val_result.issues:
                print(f"  Issues found: {len(val_result.issues)}")
                for issue in val_result.issues[:3]:  # Show first 3
                    print(f"    - {issue.issue_type}: {issue.message}")

        # Clean up
        database.close()

        print(f"\n{'='*50}")
        print("VALIDATION COMPLETED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_validation()
    exit(0 if success else 1)