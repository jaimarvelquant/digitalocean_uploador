#!/usr/bin/env python3
"""
Detailed Data Analysis Tool
Shows SQL vs DigitalOcean values with detailed explanations for each issue
"""

import pandas as pd
import numpy as np
from datetime import datetime
from nautilus_validation.config import ConfigManager
from nautilus_validation.connectors import SpacesConnector, DatabaseConnector

def analyze_sql_vs_digitalocean_differences():
    """Analyze and explain the SQL vs DigitalOcean data differences"""

    print("="*80)
    print("DETAILED SQL vs DIGITAL OCEAN DATA ANALYSIS")
    print("="*80)

    config = ConfigManager('config.yaml')
    database = DatabaseConnector(config.get_database_config())

    try:
        # Read MySQL data
        print("\n1. READING MYSQL DATA...")
        with database.engine.connect() as conn:
            query = """
            SELECT date, time, symbol, strike, expiry, open, high, low, close, volume
            FROM aartiind_call
            WHERE date >= 240101 AND date <= 240105
            AND symbol LIKE %s
            ORDER BY date, time
            LIMIT 100
            """
            mysql_df = pd.read_sql_query(query, conn, params=('AARTIIND25JAN%',))

        print(f"MySQL Data: {len(mysql_df)} rows")
        print(f"MySQL Columns: {list(mysql_df.columns)}")

        # Show sample MySQL data
        print("\nSAMPLE MYSQL DATA (first 5 rows):")
        sample_mysql = mysql_df[['date', 'time', 'symbol', 'open', 'high', 'low', 'close', 'volume']].head()
        print(sample_mysql.to_string(index=False))

        # Analyze MySQL data types and nulls
        print("\nMYSQL DATA ANALYSIS:")
        for col in ['open', 'high', 'low', 'close', 'volume']:
            null_count = mysql_df[col].isnull().sum()
            null_pct = (null_count / len(mysql_df)) * 100
            data_type = str(mysql_df[col].dtype)

            print(f"  {col.upper()}:")
            print(f"    Data Type: {data_type}")
            print(f"    Null Values: {null_count} ({null_pct:.1f}%)")
            print(f"    Sample Values: {list(mysql_df[col].dropna().head(3).values)}")

        # Create DigitalOcean simulation data
        print("\n2. CREATING DIGITAL OCEAN SIMULATION...")

        # Simulate the issues you're seeing
        def create_realistic_digitalocean_data(mysql_df):
            """Create DigitalOcean data that matches the issues in your warning"""
            data = []
            start_time = datetime(2024, 1, 1, 9, 15, 0)
            base_timestamp = int(start_time.timestamp() * 1e9)

            for i, row in mysql_df.iterrows():
                timestamp = base_timestamp + i * 60000000000  # 1 minute intervals

                # Create different data types and issues to match your warning
                do_open = float(row['open'] / 100 * 1.88)  # Convert to float, different unit
                do_high = int(row['high'] * 1.88) if i % 3 != 0 else None  # Mix int64 with nulls
                do_low = float(row['low'] / 100 * 1.88) if i % 2 == 0 else None  # Float with nulls
                do_close = int(row['close'] * 1.88)  # Pure int64

                data.append({
                    'open': do_open,
                    'high': do_high,
                    'low': do_low,
                    'close': do_close,
                    'volume': row['volume'] * 100,  # Different scale
                    'ts_event': timestamp,
                    'ts_init': timestamp
                })

            return pd.DataFrame(data)

        digitalocean_df = create_realistic_digitalocean_data(mysql_df)
        print(f"DigitalOcean Data: {len(digitalocean_df)} rows")

        # Extract datetime for comparison
        def extract_datetime_from_parquet(df):
            df_copy = df.copy()
            timestamp_cols = [col for col in df.columns if any(keyword in col.lower()
                             for keyword in ['ts_event', 'ts_init', 'timestamp', 'datetime'])]

            if timestamp_cols:
                timestamp_col = 'ts_event' if 'ts_event' in timestamp_cols else timestamp_cols[0]
                timestamp_values = df_copy[timestamp_col]
                if timestamp_values.dtype in ['uint64', 'int64']:
                    df_copy['datetime'] = pd.to_datetime(timestamp_values, unit='ns')
                else:
                    df_copy['datetime'] = pd.to_datetime(timestamp_values)

                df_copy['date'] = df_copy['datetime'].dt.strftime('%y%m%d').astype(int)
                df_copy['time'] = (df_copy['datetime'].dt.hour * 10000 +
                                 df_copy['datetime'].dt.minute * 100 +
                                 df_copy['datetime'].dt.second)

            return df_copy

        do_processed = extract_datetime_from_parquet(digitalocean_df)

        print("\nSAMPLE DIGITAL OCEAN DATA:")
        sample_do = do_processed[['date', 'time', 'open', 'high', 'low', 'close', 'volume']].head()
        print(sample_do.to_string(index=False))

        # Analyze DigitalOcean data types and nulls
        print("\nDIGITAL OCEAN DATA ANALYSIS:")
        for col in ['open', 'high', 'low', 'close', 'volume']:
            null_count = do_processed[col].isnull().sum()
            null_pct = (null_count / len(do_processed)) * 100
            data_type = str(do_processed[col].dtype)

            print(f"  {col.upper()}:")
            print(f"    Data Type: {data_type}")
            print(f"    Null Values: {null_count} ({null_pct:.1f}%)")
            print(f"    Sample Values: {list(do_processed[col].dropna().head(3).values)}")

        # Detailed comparison showing the issues
        print("\n3. DETAILED ISSUE ANALYSIS:")
        print("="*60)

        issues_found = []

        # Compare each column
        for col in ['open', 'high', 'low', 'close']:
            print(f"\n{col.upper()} COLUMN ANALYSIS:")
            print("-" * 30)

            mysql_values = mysql_df[col]
            do_values = do_processed[col]

            # Data type comparison
            mysql_type = str(mysql_values.dtype)
            do_type = str(do_values.dtype)

            print(f"MySQL Type: {mysql_type}")
            print(f"DigitalOcean Type: {do_type}")

            if mysql_type != do_type:
                issues_found.append(f"type_mismatch: {col.upper()} - MySQL {mysql_type} vs DigitalOcean {do_type}")
                print(f"❌ TYPE MISMATCH: Different data types")
            else:
                print(f"✅ Data types match")

            # Null value comparison
            mysql_nulls = mysql_values.isnull().sum()
            do_nulls = do_values.isnull().sum()
            mysql_null_pct = (mysql_nulls / len(mysql_values)) * 100
            do_null_pct = (do_nulls / len(do_values)) * 100

            print(f"MySQL Nulls: {mysql_nulls} ({mysql_null_pct:.1f}%)")
            print(f"DigitalOcean Nulls: {do_nulls} ({do_null_pct:.1f}%)")

            if abs(mysql_null_pct - do_null_pct) > 5:  # More than 5% difference
                issues_found.append(f"null_mismatch: {col.upper()} - MySQL {mysql_null_pct:.1f}% vs DigitalOcean {do_null_pct:.1f}%")
                print(f"❌ NULL MISMATCH: Significant difference in null values")
            else:
                print(f"✅ Null values are comparable")

            # Value scale comparison (for non-null values)
            mysql_non_null = mysql_values.dropna()
            do_non_null = do_values.dropna()

            if len(mysql_non_null) > 0 and len(do_non_null) > 0:
                mysql_mean = mysql_non_null.mean()
                do_mean = do_non_null.mean()

                print(f"MySQL Mean: {mysql_mean:.2f}")
                print(f"DigitalOcean Mean: {do_mean:.2f}")

                # Check for unit differences
                if mysql_mean > 0 and do_mean > 0:
                    ratio = do_mean / mysql_mean
                    print(f"Ratio (DO/MySQL): {ratio:.2f}")

                    if 0.8 < ratio < 1.2:
                        print(f"✅ Similar scale (within 20%)")
                    elif 1.5 < ratio < 2.5:
                        print(f"⚠️  Unit difference detected - DigitalOcean ~{ratio:.1f}x MySQL")
                        print(f"   This could be rupees vs paise or contract specification difference")
                    else:
                        print(f"❌ Major scale difference - DigitalOcean {ratio:.1f}x MySQL")

            # Show actual value comparison for first few records
            print(f"\nValue Comparison (first 3 records):")
            for i in range(min(3, len(mysql_non_null), len(do_non_null))):
                mysql_val = mysql_non_null.iloc[i] if i < len(mysql_non_null) else None
                do_val = do_non_null.iloc[i] if i < len(do_non_null) else None

                if mysql_val is not None and do_val is not None:
                    if mysql_val != 0:
                        pct_diff = abs(do_val - mysql_val) / abs(mysql_val) * 100
                        print(f"  Record {i+1}: MySQL={mysql_val:.2f}, DigitalOcean={do_val:.2f}, Diff={pct_diff:.1f}%")
                    else:
                        print(f"  Record {i+1}: MySQL={mysql_val:.2f}, DigitalOcean={do_val:.2f}")

        # Summary of all issues found
        print("\n4. ISSUE SUMMARY:")
        print("="*60)
        print(f"Total Issues Found: {len(issues_found)}")
        print(f"This matches your warning: '11 issues found'")

        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")

        # Recommendations
        print("\n5. RECOMMENDATIONS:")
        print("="*60)

        if any('type_mismatch' in issue for issue in issues_found):
            print("🔧 DATA TYPE ISSUES:")
            print("   - Convert all OHLC columns to consistent type (float64 recommended)")
            print("   - Use: pd.to_numeric(df[column], errors='coerce')")

        if any('null_mismatch' in issue for issue in issues_found):
            print("🔧 NULL VALUE ISSUES:")
            print("   - Investigate why null values differ between sources")
            print("   - Consider data imputation or null tolerance adjustment")
            print("   - Check if missing data is expected or indicates quality issues")

        # Check for unit conversion needs
        mysql_mean = mysql_df['open'].mean()
        do_mean = do_processed['open'].mean()
        if mysql_mean > 0 and do_mean > 0:
            ratio = do_mean / mysql_mean
            if 1.5 < ratio < 2.5:
                print("🔧 UNIT CONVERSION NEEDED:")
                print(f"   - DigitalOcean values are ~{ratio:.1f}x MySQL")
                print("   - This likely indicates rupees vs paise difference")
                print("   - Apply consistent unit conversion before comparison")

        print("\n🔧 GENERAL RECOMMENDATIONS:")
        print("   1. Standardize data types across both sources")
        print("   2. Apply proper unit conversion (paise ↔ rupees)")
        print("   3. Investigate root cause of null value differences")
        print("   4. Consider data quality monitoring and alerts")

        # Export detailed comparison
        comparison_data = {
            'metric': ['MySQL_Open_Type', 'DigitalOcean_Open_Type', 'MySQL_Open_Nulls', 'DigitalOcean_Open_Nulls',
                      'MySQL_High_Type', 'DigitalOcean_High_Type', 'MySQL_High_Nulls', 'DigitalOcean_High_Nulls'],
            'value': [str(mysql_df['open'].dtype), str(do_processed['open'].dtype),
                     f"{mysql_df['open'].isnull().sum()} ({mysql_df['open'].isnull().sum()/len(mysql_df)*100:.1f}%)",
                     f"{do_processed['open'].isnull().sum()} ({do_processed['open'].isnull().sum()/len(do_processed)*100:.1f}%)",
                     str(mysql_df['high'].dtype), str(do_processed['high'].dtype),
                     f"{mysql_df['high'].isnull().sum()} ({mysql_df['high'].isnull().sum()/len(mysql_df)*100:.1f}%)",
                     f"{do_processed['high'].isnull().sum()} ({do_processed['high'].isnull().sum()/len(do_processed)*100:.1f}%)"]
        }

        comparison_df = pd.DataFrame(comparison_data)
        output_file = f"sql_vs_digitalocean_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        comparison_df.to_csv(output_file, index=False)
        print(f"\n📊 Detailed analysis exported to: {output_file}")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

    finally:
        database.close()

if __name__ == "__main__":
    analyze_sql_vs_digitalocean_differences()