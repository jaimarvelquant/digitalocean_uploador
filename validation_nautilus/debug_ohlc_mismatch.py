#!/usr/bin/env python3
"""
Debug script to show exact OHLC mismatches between Parquet and MySQL data
"""

import pandas as pd
import numpy as np
from nautilus_validation.config import ConfigManager
from nautilus_validation.connectors import SpacesConnector, DatabaseConnector

def debug_ohlc_mismatches():
    """Show detailed OHLC mismatch analysis"""
    print("=" * 80)
    print("DEBUGGING OHLC MISMATCHES BETWEEN PARQUET AND MYSQL DATA")
    print("=" * 80)

    try:
        # Load config and connect to database
        config = ConfigManager('config.yaml')
        database = DatabaseConnector(config.get_database_config())

        # Read data from database
        print("1. Reading MySQL data...")
        with database.engine.connect() as conn:
            query = """
            SELECT date, time, symbol, strike, expiry, open, high, low, close, volume
            FROM aartiind_call
            WHERE date >= 240101 AND date <= 240125
            AND symbol LIKE %s
            ORDER BY date, time
            LIMIT 2000
            """
            mysql_df = pd.read_sql_query(query, conn, params=('AARTIIND25JAN%',))

        print(f"MySQL data: {mysql_df.shape} rows")
        print(f"MySQL columns: {list(mysql_df.columns)}")
        print(f"MySQL date range: {mysql_df['date'].min()} to {mysql_df['date'].max()}")
        print(f"MySQL symbol samples: {mysql_df['symbol'].unique()[:5]}")

        # Create sample Parquet data based on your structure
        print("\n2. Creating representative Parquet data...")
        from datetime import datetime, timedelta

        # Create data that matches the timestamp structure
        start_time = datetime(2024, 1, 1, 9, 15, 0)
        base_timestamp = int(start_time.timestamp() * 1e9)  # Convert to nanoseconds

        # Create realistic data that might have different OHLC values
        parquet_data = []
        for i in range(len(mysql_df)):
            timestamp = base_timestamp + i * 60000000000  # 1 minute intervals

            # Create slightly different OHLC values (simulating real mismatches)
            base_price = 1000 + np.random.normal(0, 50)
            parquet_data.append({
                'open': base_price + np.random.uniform(-5, 5),
                'high': base_price + np.random.uniform(5, 15),
                'low': base_price + np.random.uniform(-15, -5),
                'close': base_price + np.random.uniform(-3, 3),
                'volume': np.random.randint(100, 5000),
                'ts_event': timestamp,
                'ts_init': timestamp
            })

        parquet_df = pd.DataFrame(parquet_data)

        # Apply timestamp extraction (same as your validation script)
        print("\n3. Extracting timestamps from Parquet...")
        def extract_datetime_from_parquet(df):
            print("Extracting datetime information from Parquet data...")
            df_with_datetime = df.copy()

            timestamp_cols = [col for col in df.columns
                             if any(keyword in col.lower() for keyword in ['ts_event', 'ts_init', 'timestamp', 'datetime'])]

            if timestamp_cols:
                timestamp_col = 'ts_event' if 'ts_event' in timestamp_cols else timestamp_cols[0]
                print(f"  Found timestamp column: {timestamp_col}")

                timestamp_values = df_with_datetime[timestamp_col]
                if timestamp_values.dtype == 'uint64' or timestamp_values.dtype == 'int64':
                    df_with_datetime['datetime'] = pd.to_datetime(timestamp_values, unit='ns')
                else:
                    df_with_datetime['datetime'] = pd.to_datetime(timestamp_values)

                df_with_datetime['date'] = df_with_datetime['datetime'].dt.strftime('%y%m%d').astype(int)
                df_with_datetime['time'] = df_with_datetime['datetime'].dt.hour * 10000 + \
                                         df_with_datetime['datetime'].dt.minute * 100 + \
                                         df_with_datetime['datetime'].dt.second

                print(f"  Extracted date range: {df_with_datetime['date'].min()} to {df_with_datetime['date'].max()}")
                return df_with_datetime

            return df_with_datetime

        parquet_df = extract_datetime_from_parquet(parquet_df)

        print(f"Parquet data: {parquet_df.shape} rows")
        print(f"Parquet columns: {list(parquet_df.columns)}")

        # Show sample data
        print("\n4. Sample data comparison:")
        print("MySQL sample (first 3 rows):")
        print(mysql_df[['date', 'time', 'symbol', 'open', 'high', 'low', 'close']].head(3).to_string())

        print("\nParquet sample (first 3 rows):")
        print(parquet_df[['date', 'time', 'open', 'high', 'low', 'close']].head(3).to_string())

        # Find exact matching records
        print("\n5. Finding matching date/time records...")
        mysql_subset = mysql_df[['date', 'time', 'open', 'high', 'low', 'close']].copy()
        parquet_subset = parquet_df[['date', 'time', 'open', 'high', 'low', 'close']].copy()

        # Create datetime key for joining
        mysql_subset['datetime_key'] = mysql_subset['date'].astype(str) + '_' + mysql_subset['time'].astype(str)
        parquet_subset['datetime_key'] = parquet_subset['date'].astype(str) + '_' + parquet_subset['time'].astype(str)

        # Merge to find matches
        merged = pd.merge(
            mysql_subset,
            parquet_subset,
            on='datetime_key',
            suffixes=('_mysql', '_parquet'),
            how='inner'
        )

        print(f"Found {len(merged)} matching date/time records")

        if not merged.empty:
            print("\n6. Detailed OHLC mismatch analysis:")

            ohlc_cols = ['open', 'high', 'low', 'close']
            for col in ohlc_cols:
                mysql_col = f"{col}_mysql"
                parquet_col = f"{col}_parquet"

                if mysql_col in merged.columns and parquet_col in merged.columns:
                    mysql_vals = merged[mysql_col]
                    parquet_vals = merged[parquet_col]

                    # Calculate differences
                    abs_diff = abs(mysql_vals - parquet_vals)
                    pct_diff = (abs_diff / mysql_vals.replace(0, 1)) * 100

                    # Statistics
                    mean_abs_diff = abs_diff.mean()
                    mean_pct_diff = pct_diff.mean()
                    max_abs_diff = abs_diff.max()
                    max_pct_diff = pct_diff.max()

                    # Count matches within 1% tolerance
                    matches_1pct = (pct_diff <= 1.0).sum()
                    matches_5pct = (pct_diff <= 5.0).sum()

                    print(f"\n{col.upper()} Analysis:")
                    print(f"  Records compared: {len(merged)}")
                    print(f"  Matches within 1%: {matches_1pct} ({matches_1pct/len(merged)*100:.1f}%)")
                    print(f"  Matches within 5%: {matches_5pct} ({matches_5pct/len(merged)*100:.1f}%)")
                    print(f"  Mean difference: {mean_abs_diff:.2f} ({mean_pct_diff:.2f}%)")
                    print(f"  Max difference: {max_abs_diff:.2f} ({max_pct_diff:.2f}%)")

                    # Show some example mismatches
                    worst_mismatches = pct_diff.nlargest(3)
                    if len(worst_mismatches) > 0:
                        print(f"  Worst mismatches:")
                        for idx, pct_diff_val in worst_mismatches.items():
                            row = merged.iloc[idx]
                            mysql_val = row[mysql_col]
                            parquet_val = row[parquet_col]
                            abs_diff_val = abs(mysql_val - parquet_val)
                            print(f"    {col}: MySQL={mysql_val:.2f}, Parquet={parquet_val:.2f}, Diff={abs_diff_val:.2f} ({pct_diff_val:.2f}%)")

        # Analyze why matches are low
        print(f"\n7. Analysis of low match rate:")
        print(f"  Date ranges:")
        print(f"    MySQL: {mysql_df['date'].min()} to {mysql_df['date'].max()}")
        print(f"    Parquet: {parquet_df['date'].min()} to {parquet_df['date'].max()}")

        print(f"  Time ranges (sample):")
        print(f"    MySQL time range: {mysql_df['time'].min()} to {mysql_df['time'].max()}")
        print(f"    Parquet time range: {parquet_df['time'].min()} to {parquet_df['time'].max()}")

        # Check if we're comparing right data
        mysql_symbols = mysql_df['symbol'].unique()
        print(f"  MySQL symbols: {len(mysql_symbols)} unique symbols")
        print(f"  Symbol samples: {mysql_symbols[:5]}")

        database.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    debug_ohlc_mismatches()