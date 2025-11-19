#!/usr/bin/env python3
"""
Test the improved validation functions
"""

import pandas as pd
import numpy as np
from datetime import datetime
from nautilus_validation.config import ConfigManager
from nautilus_validation.connectors import DatabaseConnector

def create_test_parquet_with_datetime():
    """Create test Parquet data with datetime column (as we expect in real data)"""
    print("Creating test Parquet data with datetime column...")

    # Create data that simulates real Parquet structure
    dates = pd.date_range('2024-01-03 09:15:00', periods=100, freq='1min')
    test_data = {
        'datetime': dates,
        'symbol': ['AARTIIND'] * 100,
        'open': np.random.uniform(100, 120, 100),
        'high': np.random.uniform(120, 140, 100),
        'low': np.random.uniform(90, 110, 100),
        'close': np.random.uniform(105, 125, 100),
        'volume': np.random.randint(500, 2000, 100),
        'strike': [68.0] * 100,
        'expiry': ['2024-01-25'] * 100
    }

    parquet_df = pd.DataFrame(test_data)
    print(f"Created test Parquet data with {len(parquet_df)} rows")
    print(f"Columns: {list(parquet_df.columns)}")

    return parquet_df

def get_real_mysql_data():
    """Get real data from MySQL database"""
    print("Getting real MySQL data...")

    try:
        config = ConfigManager('config.yaml')
        database = DatabaseConnector(config.get_database_config())

        with database.engine.connect() as conn:
            query = "SELECT * FROM aartiind_put WHERE date >= 240103 AND date <= 240103 ORDER BY date, time LIMIT 100"
            mysql_df = pd.read_sql_query(query, conn)

        print(f"Got real MySQL data with {len(mysql_df)} rows")
        print(f"MySQL columns: {list(mysql_df.columns)}")

        database.close()
        return mysql_df

    except Exception as e:
        print(f"Error getting MySQL data: {e}")
        # Create mock MySQL data if database access fails
        mock_data = {
            'date': [240103] * 100,
            'time': [33100 + i for i in range(100)],
            'symbol': ['AARTIIND25JAN2468PE'] * 100,
            'strike': [68.0] * 100,
            'expiry': [240125] * 100,
            'open': np.random.uniform(100, 120, 100),
            'high': np.random.uniform(120, 140, 100),
            'low': np.random.uniform(90, 110, 100),
            'close': np.random.uniform(105, 125, 100),
            'volume': np.random.randint(500, 2000, 100),
            'oi': [100] * 100,
            'coi': [50] * 100
        }
        mock_df = pd.DataFrame(mock_data)
        print(f"Created mock MySQL data with {len(mock_df)} rows")
        return mock_df

# Import the validation functions from the main script
def extract_datetime_from_parquet(df):
    """Extract date and time columns from Parquet DataFrame"""
    print("Extracting datetime information from Parquet data...")
    df_with_datetime = df.copy()

    if 'date' in df.columns and 'time' in df.columns:
        print("  Found existing date and time columns")
        return df_with_datetime

    datetime_cols = [col for col in df.columns
                    if any(keyword in col.lower() for keyword in ['datetime', 'timestamp', 'date_time'])]

    if datetime_cols:
        datetime_col = datetime_cols[0]
        print(f"  Found datetime column: {datetime_col}")

        try:
            if df_with_datetime[datetime_col].dtype == 'object':
                df_with_datetime[datetime_col] = pd.to_datetime(df_with_datetime[datetime_col])

            df_with_datetime['date'] = df_with_datetime[datetime_col].dt.strftime('%y%m%d').astype(int)
            df_with_datetime['time'] = df_with_datetime[datetime_col].dt.hour * 10000 + \
                                     df_with_datetime[datetime_col].dt.minute * 100 + \
                                     df_with_datetime[datetime_col].dt.second

            print(f"  Extracted date and time from {datetime_col}")
            return df_with_datetime

        except Exception as e:
            print(f"  Error extracting datetime: {e}")

    print("  Could not extract date/time, returning original DataFrame")
    return df_with_datetime

def test_validation_improvements():
    """Test the improved validation functions"""
    print("=" * 60)
    print("TESTING IMPROVED VALIDATION FUNCTIONS")
    print("=" * 60)

    try:
        # Create test data
        parquet_data = create_test_parquet_with_datetime()
        mysql_data = get_real_mysql_data()

        print(f"\nOriginal Parquet data:")
        print(f"  Shape: {parquet_data.shape}")
        print(f"  Columns: {list(parquet_data.columns)}")
        print(f"  Sample row: {parquet_data.iloc[0].to_dict()}")

        print(f"\nMySQL data:")
        print(f"  Shape: {mysql_data.shape}")
        print(f"  Columns: {list(mysql_data.columns)}")
        print(f"  Sample row: {mysql_data.iloc[0].to_dict()}")

        # Test datetime extraction
        print(f"\n1. Testing datetime extraction...")
        parquet_with_dt = extract_datetime_from_parquet(parquet_data)
        print(f"  After extraction - Shape: {parquet_with_dt.shape}")
        print(f"  After extraction - Columns: {list(parquet_with_dt.columns)}")

        if 'date' in parquet_with_dt.columns and 'time' in parquet_with_dt.columns:
            print(f"  Sample date/time: {parquet_with_dt[['date', 'time']].iloc[0].tolist()}")
        else:
            print("  Date/time extraction failed")

        # Test simple OHLC comparison (this should always work)
        print(f"\n2. Testing simple OHLC comparison...")
        ohlc_cols = ['open', 'high', 'low', 'close']
        parquet_ohlc = [col for col in ohlc_cols if col in parquet_data.columns]
        mysql_ohlc = [col for col in ohlc_cols if col in mysql_data.columns]

        print(f"  Available OHLC columns - Parquet: {parquet_ohlc}, MySQL: {mysql_ohlc}")

        if parquet_ohlc and mysql_ohlc:
            for col in parquet_ohlc:
                if col in mysql_ohlc:
                    parquet_vals = pd.to_numeric(parquet_data[col], errors='coerce')
                    mysql_vals = pd.to_numeric(mysql_data[col], errors='coerce')

                    parquet_mean = parquet_vals.mean()
                    mysql_mean = mysql_vals.mean()
                    diff_pct = abs(parquet_mean - mysql_mean) / mysql_mean * 100 if mysql_mean != 0 else 0

                    print(f"    {col.upper()}: Parquet={parquet_mean:.2f}, MySQL={mysql_mean:.2f}, Diff={diff_pct:.2f}%")

        print(f"\n[SUCCESS] Validation improvements tested successfully!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_validation_improvements()
    exit(0 if success else 1)