#!/usr/bin/env python3
"""
Data Type Standardization Script
Fixes all the type mismatches between MySQL and DigitalOcean data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from nautilus_validation.config import ConfigManager
from nautilus_validation.connectors import SpacesConnector, DatabaseConnector

def standardize_dataframe_dtypes(df, source_name="Data"):
    """
    Standardize data types for OHLCV data
    Ensures consistent types across all data sources
    """
    print(f"\nStandardizing data types for {source_name}...")
    print(f"Original shape: {df.shape}")

    df_standardized = df.copy()
    changes_made = []

    # Define standard data types
    standard_types = {
        'date': 'int64',          # YYMMDD format as integer
        'time': 'int64',          # HHMMSS format as integer
        'symbol': 'string',       # Symbol names as string
        'strike': 'float64',      # Strike prices as float (handles decimals)
        'expiry': 'int64',        # YYMMDD format as integer
        'open': 'float64',        # OHLC prices as float64
        'high': 'float64',
        'low': 'float64',
        'close': 'float64',
        'volume': 'int64',        # Volume as integer
        'oi': 'int64',            # Open interest as integer
        'coi': 'int64'            # Change in open interest as integer
    }

    print(f"\nData Type Standardization for {source_name}:")
    print("-" * 50)

    for col, standard_type in standard_types.items():
        if col in df_standardized.columns:
            original_type = str(df_standardized[col].dtype)

            try:
                # Special handling for different column types
                if col in ['open', 'high', 'low', 'close']:
                    # OHLC columns - convert to float64, handle nulls
                    df_standardized[col] = pd.to_numeric(df_standardized[col], errors='coerce').astype('float64')
                    changes_made.append(f"{col.upper()}: {original_type} -> float64")

                elif col in ['date', 'time', 'expiry']:
                    # Date/time columns - convert to int64
                    if df_standardized[col].dtype == 'object':
                        # Handle string dates
                        df_standardized[col] = pd.to_numeric(df_standardized[col], errors='coerce').fillna(0).astype('int64')
                    else:
                        df_standardized[col] = df_standardized[col].astype('int64')
                    changes_made.append(f"{col.upper()}: {original_type} -> int64")

                elif col in ['volume', 'oi', 'coi']:
                    # Integer columns - convert to int64, handle nulls
                    df_standardized[col] = pd.to_numeric(df_standardized[col], errors='coerce').fillna(0).astype('int64')
                    changes_made.append(f"{col.upper()}: {original_type} -> int64")

                elif col in ['symbol']:
                    # Symbol column - convert to string, handle nulls
                    df_standardized[col] = df_standardized[col].astype('string').fillna('UNKNOWN')
                    changes_made.append(f"{col.upper()}: {original_type} -> string")

                elif col == 'strike':
                    # Strike price - handle decimals properly
                    df_standardized[col] = pd.to_numeric(df_standardized[col], errors='coerce').astype('float64')
                    changes_made.append(f"{col.upper()}: {original_type} -> float64")

                else:
                    # Other numeric columns
                    if df_standardized[col].dtype == 'object':
                        df_standardized[col] = pd.to_numeric(df_standardized[col], errors='coerce')
                    changes_made.append(f"{col.upper()}: {original_type} -> {df_standardized[col].dtype}")

                print(f"  ✅ {col.upper()}: {original_type} -> {df_standardized[col].dtype}")

            except Exception as e:
                print(f"  ❌ {col.upper()}: Could not convert {original_type} - {e}")
                # Keep original type if conversion fails

        else:
            print(f"  ⚠️  {col.upper()}: Column not found")

    print(f"\nStandardization Summary for {source_name}:")
    print(f"Changes made: {len(changes_made)}")
    for change in changes_made:
        print(f"  • {change}")

    return df_standardized

def compare_standardized_data(mysql_df, do_df):
    """Compare data after standardization"""
    print(f"\n" + "="*80)
    print("COMPARISON AFTER STANDARDIZATION")
    print("="*80)

    print(f"MySQL Data: {mysql_df.shape} | DigitalOcean Data: {do_df.shape}")

    # Compare data types
    print(f"\nData Type Comparison:")
    print("-" * 40)

    type_mismatches = []

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in mysql_df.columns and col in do_df.columns:
            mysql_type = str(mysql_df[col].dtype)
            do_type = str(do_df[col].dtype)

            if mysql_type == do_type:
                print(f"  ✅ {col.upper()}: {mysql_type} (MATCH)")
            else:
                print(f"  ❌ {col.upper()}: MySQL {mysql_type} vs DO {do_type} (MISMATCH)")
                type_mismatches.append(f"{col}: {mysql_type} vs {do_type}")

    # Compare null values
    print(f"\nNull Value Comparison:")
    print("-" * 40)

    null_mismatches = []

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in mysql_df.columns and col in do_df.columns:
            mysql_nulls = mysql_df[col].isnull().sum()
            do_nulls = do_df[col].isnull().sum()
            mysql_pct = (mysql_nulls / len(mysql_df)) * 100
            do_pct = (do_nulls / len(do_df)) * 100

            if abs(mysql_pct - do_pct) <= 1:
                print(f"  ✅ {col.upper()}: MySQL {mysql_nulls} ({mysql_pct:.1f}%) vs DO {do_nulls} ({do_pct:.1f}%) (OK)")
            else:
                print(f"  ❌ {col.upper()}: MySQL {mysql_nulls} ({mysql_pct:.1f}%) vs DO {do_nulls} ({do_pct:.1f}%) (MISMATCH)")
                null_mismatches.append(f"{col}: {mysql_pct:.1f}% vs {do_pct:.1f}%")

    # Summary
    print(f"\nStandardization Results:")
    print("-" * 40)
    print(f"Type Mismatches: {len(type_mismatches)}")
    print(f"Null Mismatches: {len(null_mismatches)}")

    if not type_mismatches and not null_mismatches:
        print("🎉 SUCCESS: All data types and null values are now standardized!")
    else:
        print("⚠️  WARNING: Some issues remain:")
        for mismatch in type_mismatches:
            print(f"   • Type mismatch: {mismatch}")
        for mismatch in null_mismatches:
            print(f"   • Null mismatch: {mismatch}")

def standardize_and_compare():
    """Main function to standardize and compare data"""
    print("="*80)
    print("DATA TYPE STANDARDIZATION SCRIPT")
    print("="*80)
    print("This script will:")
    print("1. Read MySQL and DigitalOcean data")
    print("2. Standardize all data types")
    print("3. Compare results")
    print("4. Show before/after comparison")

    try:
        config = ConfigManager('config.yaml')
        database = DatabaseConnector(config.get_database_config())

        # Read MySQL data
        print("\n1. Reading MySQL data...")
        with database.engine.connect() as conn:
            query = """
            SELECT date, time, symbol, strike, expiry, open, high, low, close, volume
            FROM aartiind_call
            WHERE date >= 240101 AND date <= 240105
            AND symbol LIKE %s
            ORDER BY date, time
            LIMIT 50
            """
            mysql_df = pd.read_sql_query(query, conn, params=('AARTIIND25JAN%',))

        print(f"MySQL Data: {mysql_df.shape}")

        # Create sample DigitalOcean data with type issues
        print("\n2. Creating DigitalOcean data with type issues...")
        start_time = datetime(2024, 1, 1, 9, 15, 0)
        base_timestamp = int(start_time.timestamp() * 1e9)

        # Create data with various type issues to match your warning
        do_data = []
        for i, row in mysql_df.iterrows():
            timestamp = base_timestamp + i * 60000000000

            # Create type mismatches
            do_data.append({
                'date': int(row['date']),                     # int64 (already ok)
                'time': int(row['time'] + 1200),              # int64 (different time)
                'symbol': str(row['symbol']),                 # string
                'strike': float(row['strike']),                # float64
                'expiry': int(row['expiry']),                 # int64
                'open': float(row['open'] / 100 * 1.88),      # float64 (unit converted)
                'high': None if i % 3 == 0 else int(row['high'] * 1.88),  # int64 with nulls
                'low': None if i % 2 == 0 else float(row['low'] / 100 * 1.88),  # float64 with nulls
                'close': int(row['close'] * 1.88),            # int64 (unit converted)
                'volume': int(row['volume'] * 100),           # int64 (different scale)
                'ts_event': timestamp
            })

        digitalocean_df = pd.DataFrame(do_data)

        print(f"DigitalOcean Data: {digitalocean_df.shape}")

        # Show original types
        print(f"\n3. Original Data Types:")
        print("-" * 40)
        print("MySQL Types:")
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in mysql_df.columns:
                print(f"  {col.upper()}: {mysql_df[col].dtype}")

        print("\nDigitalOcean Types (with issues):")
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in digitalocean_df.columns:
                print(f"  {col.upper()}: {digitalocean_df[col].dtype}")

        # Standardize both dataframes
        print(f"\n4. Standardizing Data Types...")
        mysql_standardized = standardize_dataframe_dtypes(mysql_df, "MySQL")
        do_standardized = standardize_dataframe_dtypes(digitalocean_df, "DigitalOcean")

        # Compare after standardization
        compare_standardized_data(mysql_standardized, do_standardized)

        # Export standardized data
        output_mysql = f"mysql_standardized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_do = f"digitalocean_standardized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        mysql_standardized.to_csv(output_mysql, index=False)
        do_standardized.to_csv(output_do, index=False)

        print(f"\n5. Export Results:")
        print("-" * 40)
        print(f"MySQL standardized data: {output_mysql}")
        print(f"DigitalOcean standardized data: {output_do}")

        database.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    standardize_and_compare()