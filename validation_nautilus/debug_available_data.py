#!/usr/bin/env python3
"""
Debug script to check available data and fix validation issues
"""

import pandas as pd
from nautilus_validation.config import ConfigManager
from nautilus_validation.connectors import SpacesConnector, DatabaseConnector

def check_available_data():
    """Check what data is actually available"""
    print("=" * 60)
    print("CHECKING AVAILABLE DATA")
    print("=" * 60)

    try:
        # Load config
        config = ConfigManager('config.yaml')

        # Check database data first
        print("1. Checking MySQL database data...")
        db_config = config.get_database_config()
        database = DatabaseConnector(db_config)

        with database.engine.connect() as conn:
            # Check the table that was mentioned in the error
            table_check_query = "SELECT COUNT(*) as row_count FROM aartiind_put"
            result = pd.read_sql_query(table_check_query, conn)
            print(f"   aartiind_put table has {result.iloc[0]['row_count']:,} rows")

            # Get sample data
            sample_query = "SELECT * FROM aartiind_put LIMIT 5"
            sample_data = pd.read_sql_query(sample_query, conn)
            print(f"   Sample columns: {list(sample_data.columns)}")
            print(f"   Sample data:")
            print(sample_data[['date', 'time', 'symbol', 'open', 'high', 'low', 'close']].to_string())

        database.close()

        # Now let's understand the Parquet data structure by creating a simple test
        print("\n2. Creating test Parquet data structure...")
        # Based on the error, it seems Parquet data might have different column structure
        # Let's create a likely scenario based on the data we see

        test_parquet_data = {
            # Maybe Parquet has datetime in a different format or single timestamp column
            'datetime': pd.date_range('2024-01-03 09:15:00', periods=3, freq='1min'),
            'symbol': ['AARTIIND', 'AARTIIND', 'AARTIIND'],
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [95.0, 96.0, 97.0],
            'close': [104.0, 105.0, 106.0],
            'volume': [1000, 1200, 800],
            'strike': [68.0, 68.0, 68.0],
            'expiry': ['2024-01-25', '2024-01-25', '2024-01-25']
        }
        test_df = pd.DataFrame(test_parquet_data)
        print(f"   Test Parquet columns: {list(test_df.columns)}")
        print(f"   Test Parquet data:")
        print(test_df.head())

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def improve_validation_functions():
    """Create improved validation functions to handle missing columns gracefully"""
    print("\n" + "=" * 60)
    print("CREATING IMPROVED VALIDATION FUNCTIONS")
    print("=" * 60)

    print("The issues identified:")
    print("1. Parquet data missing 'date' and 'time' columns")
    print("2. Need to handle different datetime formats")
    print("3. Need more robust column mapping")

    print("\nProposed solutions:")
    print("1. Detect available datetime columns in Parquet")
    print("2. Convert datetime to date/time components for comparison")
    print("3. Handle missing columns gracefully")
    print("4. Improve OHLC comparison to work with available data")

if __name__ == "__main__":
    check_available_data()
    improve_validation_functions()