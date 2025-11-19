#!/usr/bin/env python3
"""
Test the unit conversion fix for OHLC comparison
"""

import pandas as pd
import numpy as np

def test_unit_conversion():
    """Test the unit conversion functionality"""
    print("=" * 60)
    print("TESTING UNIT CONVERSION FIX")
    print("=" * 60)

    # Create test data with unit mismatch (rupees vs paise)
    # MySQL data in paise (higher values)
    mysql_data = {
        'date': [240101, 240101, 240101],
        'time': [33300, 33400, 33500],
        'symbol': ['AARTIIND25JAN700CE', 'AARTIIND25JAN700CE', 'AARTIIND25JAN700CE'],
        'open': [85000, 85500, 86000],  # 850.00 rupees = 85000 paise
        'high': [85500, 86000, 86500],  # 855.00 rupees = 85500 paise
        'low': [84500, 85000, 85500],   # 845.00 rupees = 84500 paise
        'close': [85200, 85700, 86200], # 852.00 rupees = 85200 paise
        'volume': [1000, 1200, 800]
    }

    # Parquet data in rupees (lower values)
    parquet_data = {
        'open': [850.00, 855.00, 860.00],
        'high': [855.00, 860.00, 865.00],
        'low': [845.00, 850.00, 855.00],
        'close': [852.00, 857.00, 862.00],
        'volume': [1000, 1200, 800],
        'ts_event': [1704116100000000000, 1704116160000000000, 1704116220000000000]
    }

    source_df = pd.DataFrame(parquet_data)
    target_df = pd.DataFrame(mysql_data)

    print("Original data:")
    print(f"Parquet (rupees): open={source_df['open'].iloc[0]}, high={source_df['high'].iloc[0]}")
    print(f"MySQL (paise): open={target_df['open'].iloc[0]}, high={target_df['high'].iloc[0]}")

    # Apply unit conversion logic
    def normalize_dataframe_dtypes(source_df, target_df):
        """Normalize data types and units between DataFrames"""
        print("Normalizing data types and units between datasets...")

        source_normalized = source_df.copy()
        target_normalized = target_df.copy()

        # Add timestamp extraction (simplified)
        timestamp_cols = [col for col in source_normalized.columns
                         if any(keyword in col.lower() for keyword in ['ts_event', 'timestamp'])]
        if timestamp_cols:
            timestamp_col = timestamp_cols[0]
            source_normalized['datetime'] = pd.to_datetime(source_normalized[timestamp_col], unit='ns')
            source_normalized['date'] = source_normalized['datetime'].dt.strftime('%y%m%d').astype(int)
            source_normalized['time'] = source_normalized['datetime'].dt.hour * 10000 + \
                                     source_normalized['datetime'].dt.minute * 100 + \
                                     source_normalized['datetime'].dt.second

        # Common columns
        common_columns = set(source_normalized.columns) & set(target_normalized.columns)
        ohlc_cols = [col for col in common_columns if col in ['open', 'high', 'low', 'close']]

        # Detect unit mismatch
        needs_unit_conversion = False
        if ohlc_cols:
            sample_col = ohlc_cols[0]
            source_vals = pd.to_numeric(source_normalized[sample_col], errors='coerce').dropna()
            target_vals = pd.to_numeric(target_normalized[sample_col], errors='coerce').dropna()

            if len(source_vals) > 0 and len(target_vals) > 0:
                source_mean = source_vals.mean()
                target_mean = target_vals.mean()
                ratio = target_mean / source_mean if source_mean != 0 else 1

                print(f"Unit analysis: source_mean={source_mean:.2f}, target_mean={target_mean:.2f}, ratio={ratio:.2f}")

                if 0.8 < ratio < 1.2:
                    needs_unit_conversion = False
                elif 80 < ratio < 120:
                    needs_unit_conversion = True
                    conversion_direction = 'source_to_target'
                    conversion_factor = 100
                    print(f"  Detected unit mismatch: MySQL appears to be in paise, Parquet in rupees")
                    print(f"  Will multiply Parquet values by {conversion_factor} for comparison")

        # Apply normalization
        conversion_direction = locals().get('conversion_direction')
        conversion_factor = locals().get('conversion_factor', 100)

        for col in common_columns:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                source_vals = pd.to_numeric(source_normalized[col], errors='coerce')
                target_vals = pd.to_numeric(target_normalized[col], errors='coerce')

                if needs_unit_conversion and col in ['open', 'high', 'low', 'close']:
                    if conversion_direction == 'source_to_target':
                        source_normalized[col] = source_vals * conversion_factor
                        print(f"  Normalized {col}: Converted Parquet from rupees to paise (x{conversion_factor})")
                    else:
                        source_normalized[col] = source_vals
                        target_normalized[col] = target_vals
                        print(f"  Normalized {col} to numeric (no conversion)")
                else:
                    source_normalized[col] = source_vals
                    target_normalized[col] = target_vals
                    print(f"  Normalized {col} to numeric")

        return source_normalized, target_normalized

    # Test the normalization
    norm_source, norm_target = normalize_dataframe_dtypes(source_df, target_df)

    print(f"\nAfter normalization:")
    print(f"Parquet (converted to paise): open={norm_source['open'].iloc[0]}, high={norm_source['high'].iloc[0]}")
    print(f"MySQL (paise): open={norm_target['open'].iloc[0]}, high={norm_target['high'].iloc[0]}")

    # Test OHLC comparison
    print(f"\nOHLC comparison after conversion:")
    for col in ['open', 'high', 'low', 'close']:
        source_val = norm_source[col].iloc[0]
        target_val = norm_target[col].iloc[0]
        diff = abs(source_val - target_val)
        pct_diff = (diff / target_val * 100) if target_val != 0 else 0

        status = "PASS" if pct_diff <= 1.0 else "WARN" if pct_diff <= 5.0 else "FAIL"
        print(f"  {col.upper()}: {status} - Source={source_val:.0f}, Target={target_val:.0f}, Diff={diff:.0f} ({pct_diff:.2f}%)")

    return True

if __name__ == "__main__":
    test_unit_conversion()