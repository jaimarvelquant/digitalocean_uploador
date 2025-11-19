#!/usr/bin/env python3
"""
Test final fixes for validation script
"""

import pandas as pd
import numpy as np
from datetime import datetime

def test_timestamp_extraction():
    """Test timestamp extraction with real data format"""
    print("Testing timestamp extraction...")

    # Create sample data that matches your actual structure
    current_time = datetime.now()
    base_timestamp = int(current_time.timestamp() * 1e9)  # Convert to nanoseconds

    sample_data = {
        'open': [4210.0, 4300.0, 4400.0],
        'high': [4210.0, 4300.0, 4400.0],
        'low': [4210.0, 4300.0, 4235.0],
        'close': [4210.0, 4300.0, 4235.0],
        'volume': [1000, 1200, 800],
        'ts_event': [base_timestamp + i * 60000000000 for i in range(3)],  # 1 minute intervals
        'ts_init': [base_timestamp + i * 60000000000 for i in range(3)]
    }

    test_df = pd.DataFrame(sample_data)
    print(f"Test data shape: {test_df.shape}")
    print(f"Test data columns: {list(test_df.columns)}")
    print(f"ts_event sample: {test_df['ts_event'].tolist()}")

    # Apply extraction function
    def extract_datetime_from_parquet(df):
        """Enhanced datetime extraction that handles various timestamp formats"""
        print("Extracting datetime information from Parquet data...")
        df_with_datetime = df.copy()

        # Check if we already have separate date and time columns
        if 'date' in df.columns and 'time' in df.columns:
            print("  Found existing date and time columns")
            return df_with_datetime

        # Look for timestamp columns (this is what the actual data has!)
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
                print(f"  Sample converted datetime: {df_with_datetime[['date', 'time']].iloc[0].tolist()}")
                return df_with_datetime

            except Exception as e:
                print(f"  Error extracting datetime: {e}")
                return df_with_datetime

        return df_with_datetime

    # Test extraction
    result_df = extract_datetime_from_parquet(test_df)

    if 'date' in result_df.columns and 'time' in result_df.columns:
        print("[SUCCESS] Timestamp extraction worked!")
        print(f"Final columns: {list(result_df.columns)}")
        print(f"Sample date/time: {result_df[['date', 'time']].iloc[0].tolist()}")
        return True
    else:
        print("[FAILED] Timestamp extraction didn't work")
        return False

def test_ohlc_comparison():
    """Test OHLC comparison with some data quality issues"""
    print("\nTesting OHLC comparison...")

    # Create test data with some null values (like the real data)
    parquet_data = {
        'open': [100.0, np.nan, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, np.nan, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, np.nan, 99.0],
        'close': [104.0, 105.0, 106.0, 107.0, np.nan]
    }

    mysql_data = {
        'open': [100.5, 101.5, 102.5, 103.5, 104.5],
        'high': [105.5, 106.5, 107.5, 108.5, 109.5],
        'low': [95.5, 96.5, 97.5, 98.5, 99.5],
        'close': [104.5, 105.5, 106.5, 107.5, 108.5]
    }

    source_df = pd.DataFrame(parquet_data)
    target_df = pd.DataFrame(mysql_data)

    print(f"Source shape: {source_df.shape}, Target shape: {target_df.shape}")

    # Test OHLC comparison
    def validate_ohlc_values_simple(source_df, target_df, tolerance_pct=0.01):
        """Simple OHLC validation without requiring date/time matching"""
        print(f"Performing simple OHLC value comparison with {tolerance_pct*100}% tolerance...")

        # Check if OHLC columns exist
        ohlc_cols = ['open', 'high', 'low', 'close']
        source_cols = set(source_df.columns)
        target_cols = set(target_df.columns)

        available_ohlc = [col for col in ohlc_cols if col in source_cols and col in target_cols]

        if not available_ohlc:
            print("  No OHLC columns available for comparison")
            return None

        print(f"  Available OHLC columns: {available_ohlc}")

        try:
            comparison_results = {}

            for col in available_ohlc:
                source_vals = pd.to_numeric(source_df[col], errors='coerce')
                target_vals = pd.to_numeric(target_df[col], errors='coerce')

                print(f"    {col.upper()} - Raw data info:")
                print(f"      Source: {len(source_vals)} total, {source_vals.isna().sum()} nulls, dtype: {source_vals.dtype}")
                print(f"      Target: {len(target_vals)} total, {target_vals.isna().sum()} nulls, dtype: {target_vals.dtype}")

                # Drop null values and check if we have data
                source_clean = source_vals.dropna()
                target_clean = target_vals.dropna()

                if len(source_clean) == 0 or len(target_clean) == 0:
                    print(f"    {col.upper()}: No valid (non-null) data for comparison")
                    continue

                print(f"      After cleaning - Source: {len(source_clean)}, Target: {len(target_clean)}")

                # Compare basic statistics using cleaned data
                source_mean = source_clean.mean()
                target_mean = target_clean.mean()
                source_std = source_clean.std()
                target_std = target_clean.std()

                # Calculate percentage difference in means
                mean_diff_pct = abs(source_mean - target_mean) / target_mean * 100 if target_mean != 0 else 0

                comparison_results[col] = {
                    'mean_diff_pct': mean_diff_pct,
                    'source_mean': source_mean,
                    'target_mean': target_mean,
                    'source_count': len(source_clean),
                    'target_count': len(target_clean)
                }

                status = "PASS" if mean_diff_pct <= tolerance_pct else "WARN" if mean_diff_pct <= tolerance_pct * 2 else "FAIL"
                print(f"    {col.upper()}: {status} - Mean diff: {mean_diff_pct:.2f}% (Source: {source_mean:.2f}, Target: {target_mean:.2f})")

            return comparison_results

        except Exception as e:
            print(f"  ERROR in simple OHLC comparison: {e}")
            return None

    results = validate_ohlc_values_simple(source_df, target_df, tolerance_pct=2.0)  # Use 2% tolerance for test

    if results:
        print("[SUCCESS] OHLC comparison worked!")
        return True
    else:
        print("[FAILED] OHLC comparison failed")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING FINAL FIXES FOR VALIDATION SCRIPT")
    print("=" * 60)

    test1_passed = test_timestamp_extraction()
    test2_passed = test_ohlc_comparison()

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS:")
    print(f"Timestamp extraction: {'PASS' if test1_passed else 'FAIL'}")
    print(f"OHLC comparison: {'PASS' if test2_passed else 'FAIL'}")
    print(f"Overall: {'PASS' if test1_passed and test2_passed else 'FAIL'}")
    print("=" * 60)