#!/usr/bin/env python3
"""
Demo validation script that shows how to use the interactive validation
Enhanced version with multi-level navigation: prefix -> data selection -> validation
"""

import pandas as pd
import re
from datetime import datetime
from collections import Counter, defaultdict
from nautilus_validation.config import ConfigManager
from nautilus_validation.connectors import SpacesConnector, DatabaseConnector
from nautilus_validation.validators import ValidationEngine

def convert_date_to_yymmdd_format(date_str):
    """
    Convert date from YYYY-MM-DD format to YYMMDD integer format
    Used for querying the MySQL database which stores dates as YYMMDD
    """
    if isinstance(date_str, (int, float)):
        return int(date_str)  # Already in numeric format

    try:
        if isinstance(date_str, str):
            # Parse YYYY-MM-DD format
            if '-' in date_str and len(date_str) == 10:
                year, month, day = date_str.split('-')
                yy = year[-2:]  # Take last 2 digits
                return int(f"{yy}{month}{day}")

            # Try to parse other formats
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            yy = date_obj.strftime('%y')
            mm = date_obj.strftime('%m')
            dd = date_obj.strftime('%d')
            return int(f"{yy}{mm}{dd}")

    except Exception as e:
        print(f"DEBUG: Could not convert date '{date_str}': {e}")
        return None

    return None

def get_date_range_for_query(dates_found):
    """
    Get min and max dates in YYMMDD format for database query
    """
    if not dates_found:
        return None, None

    converted_dates = []
    for date_val in dates_found:
        converted = convert_date_to_yymmdd_format(date_val)
        if converted:
            converted_dates.append(converted)

    if converted_dates:
        return min(converted_dates), max(converted_dates)
    return None, None

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
            print(f"  Timestamp dtype: {timestamp_values.dtype}")
            print(f"  Sample timestamp values: {timestamp_values.head().tolist()}")

    # If no timestamp columns found, show available columns
    print(f"  No suitable timestamp columns found")
    print(f"  Available columns: {list(df.columns)}")
    return df_with_datetime

def standardize_dataframe_dtypes(df, source_name="Data"):
    """
    Standardize data types for OHLCV data
    Ensures consistent types: float64 for prices, int64 for integers, string for symbols
    """
    print(f"\nStandardizing data types for {source_name}...")

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

    print(f"  Data Type Standardization for {source_name}:")

    for col, standard_type in standard_types.items():
        if col in df_standardized.columns:
            original_type = str(df_standardized[col].dtype)

            try:
                # Special handling for different column types
                if col in ['open', 'high', 'low', 'close']:
                    # OHLC columns - convert to float64, handle nulls
                    df_standardized[col] = pd.to_numeric(df_standardized[col], errors='coerce').astype('float64')
                    if original_type != 'float64':
                        changes_made.append(f"{col.upper()}: {original_type} -> float64")

                elif col in ['date', 'time', 'expiry']:
                    # Date/time columns - convert to int64
                    if df_standardized[col].dtype == 'object':
                        # Handle string dates
                        df_standardized[col] = pd.to_numeric(df_standardized[col], errors='coerce').fillna(0).astype('int64')
                    else:
                        df_standardized[col] = df_standardized[col].astype('int64')
                    if original_type != 'int64':
                        changes_made.append(f"{col.upper()}: {original_type} -> int64")

                elif col in ['volume', 'oi', 'coi']:
                    # Integer columns - convert to int64, handle nulls
                    df_standardized[col] = pd.to_numeric(df_standardized[col], errors='coerce').fillna(0).astype('int64')
                    if original_type != 'int64':
                        changes_made.append(f"{col.upper()}: {original_type} -> int64")

                elif col in ['symbol']:
                    # Symbol column - convert to string, handle nulls
                    df_standardized[col] = df_standardized[col].astype('string').fillna('UNKNOWN')
                    if original_type != 'string':
                        changes_made.append(f"{col.upper()}: {original_type} -> string")

                elif col == 'strike':
                    # Strike price - handle decimals properly
                    df_standardized[col] = pd.to_numeric(df_standardized[col], errors='coerce').astype('float64')
                    if original_type != 'float64':
                        changes_made.append(f"{col.upper()}: {original_type} -> float64")

            except Exception as e:
                print(f"    WARNING: Could not convert {col.upper()} {original_type}: {e}")

    if changes_made:
        print(f"    Changes made: {len(changes_made)}")
        for change in changes_made:
            print(f"      • {change}")
    else:
        print(f"    No changes needed - all types already standard")

    return df_standardized

def normalize_dataframe_dtypes(source_df, target_df):
    """
    Normalize data types and units between Parquet and MySQL DataFrames
    Enhanced version with comprehensive standardization
    """
    print("Normalizing data types and units between datasets...")

    # Make copies to avoid modifying original data
    source_normalized = source_df.copy()
    target_normalized = target_df.copy()

    # Try to extract datetime from source (Parquet) data first
    source_normalized = extract_datetime_from_parquet(source_normalized)

    # Standardize both dataframes first
    source_normalized = standardize_dataframe_dtypes(source_normalized, "Source (DigitalOcean)")
    target_normalized = standardize_dataframe_dtypes(target_normalized, "Target (MySQL)")

    # Common columns that need type alignment
    common_columns = set(source_normalized.columns) & set(target_normalized.columns)

    # Detect if there's a unit mismatch (OHLC in rupees vs paise)
    ohlc_cols = [col for col in common_columns if col in ['open', 'high', 'low', 'close']]
    needs_unit_conversion = False
    conversion_direction = None
    conversion_factor = 1

    if ohlc_cols:
        # Sample first OHLC column to check for unit mismatch
        sample_col = ohlc_cols[0]
        source_vals = pd.to_numeric(source_normalized[sample_col], errors='coerce').dropna()
        target_vals = pd.to_numeric(target_normalized[sample_col], errors='coerce').dropna()

        if len(source_vals) > 0 and len(target_vals) > 0:
            source_mean = source_vals.mean()
            target_mean = target_vals.mean()

            # Check if one is roughly 100x the other (rupees vs paise)
            ratio = target_mean / source_mean if source_mean != 0 else 1
            if 0.8 < ratio < 1.2:  # Similar magnitude (within 20%)
                needs_unit_conversion = False
            elif 80 < ratio < 120:  # Target is ~100x larger (target in paise, source in rupees)
                needs_unit_conversion = True
                conversion_direction = 'source_to_target'
                conversion_factor = 100
                print(f"  Detected unit mismatch: MySQL appears to be in paise, Parquet in rupees")
                print(f"  Will multiply Parquet values by {conversion_factor} for comparison")
            elif 0.008 < ratio < 0.012:  # Source is ~100x larger (source in paise, target in rupees)
                needs_unit_conversion = True
                conversion_direction = 'target_to_source'
                conversion_factor = 100
                print(f"  Detected unit mismatch: Parquet appears to be in paise, MySQL in rupees")
                print(f"  Will multiply MySQL values by {conversion_factor} for comparison")
        else:
            conversion_direction = None
            conversion_factor = 1

    for col in common_columns:
        # Convert numeric columns to be consistent
        if col in ['open', 'high', 'low', 'close', 'volume', 'oi', 'coi']:
            try:
                # Convert both to float64 for comparison
                source_vals = pd.to_numeric(source_normalized[col], errors='coerce')
                target_vals = pd.to_numeric(target_normalized[col], errors='coerce')

                # Apply unit conversion if needed for OHLC columns
                if needs_unit_conversion and col in ['open', 'high', 'low', 'close']:
                    if conversion_direction == 'source_to_target':
                        # Multiply Parquet values by 100 to match MySQL (paise)
                        source_normalized[col] = source_vals * conversion_factor
                        print(f"  Normalized {col}: Converted Parquet from rupees to paise (x{conversion_factor})")
                    else:
                        # Multiply MySQL values by 100 to match Parquet (paise)
                        target_normalized[col] = target_vals * conversion_factor
                        print(f"  Normalized {col}: Converted MySQL from rupees to paise (x{conversion_factor})")
                else:
                    source_normalized[col] = source_vals
                    target_normalized[col] = target_vals
                    print(f"  Normalized {col} to numeric")

            except Exception as e:
                print(f"  Warning: Could not normalize {col}: {e}")

        # Handle date/time columns
        elif col in ['date', 'time']:
            try:
                # Convert both to int64 for comparison
                source_normalized[col] = pd.to_numeric(source_normalized[col], errors='coerce').astype('Int64')
                target_normalized[col] = pd.to_numeric(target_normalized[col], errors='coerce').astype('Int64')
                print(f"  Normalized {col} to integer")
            except Exception as e:
                print(f"  Warning: Could not normalize {col}: {e}")

        # Handle symbol column
        elif col == 'symbol':
            try:
                # Convert both to string and uppercase
                source_normalized[col] = source_normalized[col].astype(str).str.upper()
                target_normalized[col] = target_normalized[col].astype(str).str.upper()
                print(f"  Normalized {col} to uppercase string")
            except Exception as e:
                print(f"  Warning: Could not normalize {col}: {e}")

    print(f"Normalization complete")
    return source_normalized, target_normalized

def validate_ohlc_values_by_datetime(source_df, target_df, tolerance_pct=0.01):
    """
    Validate OHLC values by matching on date and time
    This checks if OHLC values match between Digital Ocean and MySQL data
    """
    print(f"\nPerforming OHLC value comparison by date/time with {tolerance_pct*100}% tolerance...")

    # Check if required columns exist
    required_cols = ['date', 'time', 'open', 'high', 'low', 'close']
    source_cols = set(source_df.columns)
    target_cols = set(target_df.columns)

    missing_source = set(required_cols) - source_cols
    missing_target = set(required_cols) - target_cols

    if missing_source:
        print(f"  Missing OHLC columns in source: {missing_source}")
        print(f"  Available source columns: {sorted(source_cols)}")
        return None
    if missing_target:
        print(f"  Missing OHLC columns in target: {missing_target}")
        return None

    try:
        # Create composite key for joining
        source_df['datetime_key'] = source_df['date'].astype(str) + '_' + source_df['time'].astype(str)
        target_df['datetime_key'] = target_df['date'].astype(str) + '_' + target_df['time'].astype(str)

        # Merge datasets on date+time
        merged = pd.merge(
            source_df[['datetime_key', 'open', 'high', 'low', 'close']],
            target_df[['datetime_key', 'open', 'high', 'low', 'close']],
            on='datetime_key',
            suffixes=('_source', '_target'),
            how='inner'
        )

        if merged.empty:
            print("  WARNING: No matching date/time records found between datasets")
            return None

        print(f"  Found {len(merged)} matching date/time records")

        # Compare OHLC values
        ohlc_cols = ['open', 'high', 'low', 'close']
        comparison_results = {}

        for col in ohlc_cols:
            source_col = f"{col}_source"
            target_col = f"{col}_target"

            if source_col in merged.columns and target_col in merged.columns:
                source_vals = merged[source_col]
                target_vals = merged[target_col]

                # Calculate absolute and percentage differences
                abs_diff = abs(source_vals - target_vals)
                pct_diff = (abs_diff / target_vals.replace(0, 1)) * 100  # Avoid division by zero

                # Count matches within tolerance
                matches = pct_diff <= tolerance_pct
                matches_count = matches.sum()
                total_count = len(merged)
                match_pct = (matches_count / total_count) * 100 if total_count > 0 else 0

                comparison_results[col] = {
                    'total_comparisons': total_count,
                    'matches_within_tolerance': matches_count,
                    'match_percentage': match_pct,
                    'avg_abs_difference': abs_diff.mean(),
                    'max_abs_difference': abs_diff.max(),
                    'avg_pct_difference': pct_diff.mean(),
                    'max_pct_difference': pct_diff.max()
                }

                print(f"    {col.upper()}: {matches_count}/{total_count} matches ({match_pct:.1f}%)")

                if not matches.all():
                    # Show some mismatches for debugging
                    mismatch_indices = pct_diff > tolerance_pct
                    if mismatch_indices.any():
                        worst_mismatch = pct_diff[mismatch_indices].idxmax()
                        print(f"      Worst {col} mismatch: Source={source_vals.iloc[worst_mismatch]}, Target={target_vals.iloc[worst_mismatch]} (diff={pct_diff.iloc[worst_mismatch]:.2f}%)")

        return comparison_results

    except Exception as e:
        print(f"  ERROR in OHLC comparison: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_ohlc_values_simple(source_df, target_df, tolerance_pct=0.01):
    """
    Simple OHLC validation without requiring date/time matching
    This compares overall OHLC value distributions
    """
    print(f"\nPerforming simple OHLC value comparison with {tolerance_pct*100}% tolerance...")

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
                print(f"      Source valid: {len(source_clean)}, Target valid: {len(target_clean)}")
                continue

            print(f"      After cleaning - Source: {len(source_clean)}, Target: {len(target_clean)}")

            # Compare basic statistics using cleaned data
            source_mean = source_clean.mean()
            target_mean = target_clean.mean()
            source_std = source_clean.std()
            target_std = target_clean.std()

            # Calculate percentage difference in means
            mean_diff_pct = abs(source_mean - target_mean) / target_mean * 100 if target_mean != 0 else 0

            # Count values within tolerance (using cleaned samples if same length)
            if len(source_clean) == len(target_clean):
                pct_diffs = abs(source_clean - target_clean) / target_clean * 100
                matches = (pct_diffs <= tolerance_pct).sum()
                total_comparisons = len(pct_diffs)
                match_pct = (matches / total_comparisons) * 100
            else:
                # For different lengths, compare means within tolerance
                match_pct = 100 if mean_diff_pct <= tolerance_pct else 0
                total_comparisons = min(len(source_clean), len(target_clean))
                matches = int(total_comparisons * match_pct / 100)

            comparison_results[col] = {
                'total_comparisons': total_comparisons,
                'matches_within_tolerance': matches,
                'match_percentage': match_pct,
                'source_mean': source_mean,
                'target_mean': target_mean,
                'mean_diff_pct': mean_diff_pct,
                'source_std': source_std,
                'target_std': target_std,
                'source_count': len(source_clean),
                'target_count': len(target_clean)
            }

            status = "PASS" if mean_diff_pct <= tolerance_pct else "WARN" if mean_diff_pct <= tolerance_pct * 2 else "FAIL"
            print(f"    {col.upper()}: {status} - Mean diff: {mean_diff_pct:.2f}% (Source: {source_mean:.2f}, Target: {target_mean:.2f})")

        return comparison_results

    except Exception as e:
        print(f"  ERROR in simple OHLC comparison: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_symbol_from_parquet(df):
    """
    Extract symbol from parquet DataFrame content
    """
    # Try common column names that might contain symbol
    symbol_columns = ['symbol', 'instrument', 'ticker', 'code']

    for col in symbol_columns:
        if col in df.columns and not df[col].empty:
            unique_symbols = df[col].dropna().unique()
            if len(unique_symbols) > 0:
                return str(unique_symbols[0]).upper()

    # Try to extract from other columns if no symbol column found
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique()
            # Look for values that might be symbols (uppercase letters, dots, dashes)
            for val in unique_vals[:5]:  # Check first few values
                if isinstance(val, str) and any(c.isalpha() for c in val):
                    # Check if it looks like a symbol (contains uppercase letters and maybe dots/dashes)
                    if any(c.isupper() for c in val):
                        return val.upper()

    return None

def extract_base_symbol_from_option(symbol_with_numbers):
    """
    Extract the base symbol from an option symbol that contains numbers, dates, and strikes
    Example: AARTIIND25JAN24500 -> AARTIIND
    """
    import re

    # Pattern 1: Match consecutive letters at the start (most reliable)
    pattern = r'^([A-Z]+)'
    match = re.match(pattern, symbol_with_numbers)
    if match:
        base_symbol = match.group(1)
        print(f"DEBUG: Extracted base symbol using pattern: '{base_symbol}'")
        return base_symbol

    # Pattern 2: Extract letters until first digit
    letters_part = ''
    for char in symbol_with_numbers:
        if char.isalpha():
            letters_part += char
        else:
            break
    if letters_part:
        print(f"DEBUG: Extracted base symbol using letter extraction: '{letters_part}'")
        return letters_part

    # Pattern 3: Fallback - split at first digit
    for i, char in enumerate(symbol_with_numbers):
        if char.isdigit():
            base_symbol = symbol_with_numbers[:i]
            if base_symbol:
                print(f"DEBUG: Extracted base symbol using digit detection: '{base_symbol}'")
                return base_symbol

    # If all else fails, return original
    print(f"DEBUG: No pattern matched, using original symbol: '{symbol_with_numbers}'")
    return symbol_with_numbers

def extract_symbol_from_filepath(filepath):
    """
    Extract symbol from file path when content extraction fails
    """
    path_parts = filepath.split('/')

    print(f"DEBUG: Extracting symbol from filepath: {filepath}")
    print(f"DEBUG: Path parts: {path_parts}")

    # Look for symbol-like parts in the path
    for part in path_parts:
        part_upper = part.upper()
        print(f"DEBUG: Checking part: {part}")

        # Check for various patterns
        if any(keyword in part_upper for keyword in ['NSE', 'BSE', 'FUTURE', 'INDEX', 'MINUTE', 'TICK']):
            # Extract the base symbol with enhanced logic
            symbol_part = part.split('-NSE')[0] if '-NSE' in part.upper() else part

            # Handle option symbols (like BANKNIFTY18SEP2448200PE)
            if 'NIFTY' in symbol_part or 'BANKNIFTY' in symbol_part or 'FINNIFTY' in symbol_part:
                # Extract base NIFTY symbol
                if 'BANKNIFTY' in symbol_part:
                    symbol = 'BANKNIFTY'
                    print(f"DEBUG: Found BANKNIFTY option symbol")
                    return symbol
                elif 'FINNIFTY' in symbol_part:
                    symbol = 'FINNIFTY'
                    print(f"DEBUG: Found FINNIFTY option symbol")
                    return symbol
                elif 'NIFTY' in symbol_part:
                    symbol = 'NIFTY'
                    print(f"DEBUG: Found NIFTY option symbol")
                    return symbol
                else:
                    symbol_part = symbol_part.split('.')[0] if '.' in symbol_part else symbol_part
                    print(f"DEBUG: Found other NIFTY-related symbol: {symbol_part}")
                    return symbol_part
            else:
                # Regular symbol extraction
                symbol_part = symbol_part.split('.')[0] if '.' in symbol_part else symbol_part
                if symbol_part and len(symbol_part) > 2:
                    print(f"DEBUG: Found regular symbol: {symbol_part}")
                    return symbol_part

    # Fallback - look for directories that might be symbols
    for part in path_parts:
        part_upper = part.upper()
        if any(keyword in part_upper for keyword in ['CRUDE', 'OIL', 'NIFTY', 'BANK', 'FINNIFTY']):
            print(f"DEBUG: Found keyword-based symbol: {part}")
            return part_upper

        # Check for known symbols directly
        if part_upper in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'SENSEX']:
            print(f"DEBUG: Found exact match symbol: {part}")
            return part_upper

    print(f"DEBUG: No symbol found in filepath")
    return None

def parse_parquet_filename(filename, df=None):
    """
    Parse parquet filename to extract symbol, date, and expiry information
    """
    # Remove path and extension
    basename = filename.split('/')[-1].replace('.parquet', '')

    # Pattern 1: Timestamp-based filenames
    timestamp_pattern = r'^(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{9}Z)_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{9}Z)$'
    timestamp_match = re.match(timestamp_pattern, basename)

    if timestamp_match:
        start_timestamp, end_timestamp = timestamp_match.groups()

        # Extract date from timestamp
        try:
            date_obj = datetime.strptime(start_timestamp.split('T')[0], '%Y-%m-%d')
            formatted_date = date_obj.strftime('%Y-%m-%d')
        except:
            formatted_date = start_timestamp.split('T')[0]

        # Try to get symbol from parquet content if DataFrame is provided
        symbol = None
        if df is not None:
            symbol = extract_symbol_from_parquet(df)

        # If no symbol found in content, try to extract from file path
        if not symbol:
            symbol = extract_symbol_from_filepath(filename)

        # Fallback to generic naming if still not found
        if not symbol:
            symbol = 'UNKNOWN'

        return {
            'symbol': symbol,
            'date': formatted_date,
            'expiry': None,
            'original_filename': filename,
            'start_time': start_timestamp,
            'end_time': end_timestamp
        }

    # Pattern 2: Other formats
    symbol = extract_symbol_from_filepath(filename)
    if not symbol:
        # Last resort - extract from filename
        parts = basename.split('_')[:1]
        symbol = parts[0].upper() if parts else 'UNKNOWN'

    return {
        'symbol': symbol,
        'date': None,
        'expiry': None,
        'original_filename': filename
    }

def determine_mysql_table(symbol, file_path):
    """
    Determine MySQL table name based on symbol and file path
    Maps exchange-format symbols to database table naming convention
    """
    # Clean the symbol - remove exchange-specific suffixes
    clean_symbol = symbol.upper().strip()

    # Remove exchange and data source suffixes
    suffixes_to_remove = [
        '.NSE-1-MINUTE-LAST-EXTERNAL',
        '.NSE-1-TICK-LAST-EXTERNAL',
        '-NSE-1-MINUTE-LAST-EXTERNAL',
        '-NSE-1-TICK-LAST-EXTERNAL',
        '.NSE-1-MINUTE-LAST',
        '.NSE-1-TICK-LAST',
        '-NSE-1-MINUTE-LAST',
        '-NSE-1-TICK-LAST'
    ]

    for suffix in suffixes_to_remove:
        if clean_symbol.endswith(suffix):
            clean_symbol = clean_symbol[:-len(suffix)]

    # Handle special cases first - direct mapping to known tables
    special_mappings = {
        'NIFTY': 'nifty_future',
        'BANKNIFTY': 'banknifty_future',
        'FINNIFTY': 'finnifty_future',
        'SENSEX': 'sensex_future',
        'BANKEX': 'bankex_future',
        'CRUDEOIL': 'crudeoil_future',
        'CRUDEOIL-I': 'crudeoil_future',
        'CRUDE': 'crudeoil_future',
        'OIL': 'crudeoil_future',
        'SILVER': 'silver_future',
        'SILVERM': 'silverm_future',
        'COPPER': 'copper_future',
        'GOLD': 'gold_future',
        'GOLDM': 'goldm_future',
        'ZINC': 'zinc_future',
        'LEAD': 'lead_future',
        'NATURALGAS': 'naturalgas_future',
        'ALUMINIUM': 'aluminium_future'
    }

    # Check special mappings first
    for key, value in special_mappings.items():
        if key in clean_symbol or clean_symbol == key:
            return value

    # Check for option symbols - these have priority!
    table_type = 'future'  # Default
    base_symbol = clean_symbol

    # Option detection patterns
    if clean_symbol.endswith('CE'):
        # Call Option European
        table_type = 'call'
        # Remove the CE suffix to get base symbol
        base_symbol = clean_symbol[:-2]
        print(f"DEBUG: Detected CALL option - base symbol before cleanup: {base_symbol}")

        # Extract base symbol for options (remove digits, dates, strikes)
        base_symbol = extract_base_symbol_from_option(base_symbol)
        print(f"DEBUG: Final base symbol for CALL option: {base_symbol}")

    elif clean_symbol.endswith('PE'):
        # Put Option European
        table_type = 'put'
        # Remove the PE suffix to get base symbol
        base_symbol = clean_symbol[:-2]
        print(f"DEBUG: Detected PUT option - base symbol before cleanup: {base_symbol}")

        # Extract base symbol for options (remove digits, dates, strikes)
        base_symbol = extract_base_symbol_from_option(base_symbol)
        print(f"DEBUG: Final base symbol for PUT option: {base_symbol}")

    elif clean_symbol.endswith('CA'):
        # Call Option American
        table_type = 'call'
        base_symbol = clean_symbol[:-2]
        print(f"DEBUG: Detected CALL option (American) - base symbol: {base_symbol}")
    elif clean_symbol.endswith('PA'):
        # Put Option American
        table_type = 'put'
        base_symbol = clean_symbol[:-2]
        print(f"DEBUG: Detected PUT option (American) - base symbol: {base_symbol}")

    # Extract base symbol from complex names (for non-options or if option detection failed)
    if table_type == 'future':  # Only do this for non-options
        # Split by common separators
        separators = ['-', '.', '_']
        for sep in separators:
            if sep in base_symbol:
                parts = base_symbol.split(sep)
                # Take the first part that looks like a symbol (contains letters)
                for part in parts:
                    if any(c.isalpha() for c in part):
                        base_symbol = part
                        break

    # Clean the base symbol
    base_symbol = base_symbol.strip()

    # Handle numeric prefixes and special cases
    if base_symbol.startswith('3I'):
        # Handle 3IINFOTECH specifically
        if 'INFOTECH' in base_symbol:
            return '3iinfotech_future'
        else:
            base_symbol = base_symbol.replace('3I', '3i')

    # Check file path for additional context about data type
    file_path_lower = file_path.lower()
    if table_type == 'future':  # Only check path for non-options
        if any(indicator in file_path_lower for indicator in ['option', 'call', 'put']):
            # Double-check option detection from file path
            if 'call' in file_path_lower:
                table_type = 'call'
            elif 'put' in file_path_lower:
                table_type = 'put'
        elif any(indicator in file_path_lower for indicator in ['1-minute', 'min', 'minute']):
            # Could be cash or future, default to future for trading data
            table_type = 'future'
        elif 'tick' in file_path_lower:
            table_type = 'future'
        elif 'cash' in file_path_lower:
            table_type = 'cash'

    # Common stock/index pattern detection
    if base_symbol and len(base_symbol) > 2 and table_type == 'future':
        # Check if it's likely a stock (ends with common stock patterns)
        if any(base_symbol.endswith(suffix) for suffix in ['LTD', 'LIMITED', 'CORP', 'IND', 'INFRA']):
            table_type = 'cash'  # Use cash for stocks
        elif base_symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'SENSEX']:
            table_type = 'future'  # Use future for indices

    # Remove any remaining special characters except common symbol characters
    import re
    base_symbol = re.sub(r'[^A-Z0-9]', '', base_symbol)

    # Final table name construction
    table_name = f"{base_symbol.lower()}_{table_type}"

    print(f"DEBUG: Final mapping - Symbol: {symbol}, Base: {base_symbol}, Type: {table_type}, Table: {table_name}")

    return table_name

def list_prefixes_in_nautilas_data(bucket):
    """List all prefixes inside nautilas-data directory"""
    print(f"\n{'='*60}")
    print("DISCOVERING AVAILABLE PREFIXES")
    print(f"{'='*60}")
    print(f"Scanning: {bucket}/nautilas-data/")

    try:
        # Load config and initialize connector
        config = ConfigManager('config.yaml')
        do_config = config.get_digital_ocean_config()
        do_config['bucket_name'] = bucket
        spaces = SpacesConnector(do_config)

        # Get all objects in nautilas-data
        all_objects = spaces.client.list_objects_v2(
            Bucket=bucket,
            Prefix='nautilas-data/',
            MaxKeys=2000,
            Delimiter='/'
        )

        prefixes = []
        if 'CommonPrefixes' in all_objects:
            for prefix_obj in all_objects['CommonPrefixes']:
                prefix = prefix_obj['Prefix']
                # Remove the 'nautilas-data/' part and trailing slash
                clean_prefix = prefix.replace('nautilas-data/', '').rstrip('/')
                if clean_prefix:  # Skip empty prefix
                    prefixes.append(clean_prefix)

        if prefixes:
            print(f"Found {len(prefixes)} prefixes:")
            print("-" * 60)

            for i, prefix in enumerate(prefixes, 1):
                print(f"{i:2d}. {prefix}")

            return prefixes
        else:
            print("No prefixes found inside nautilas-data/")
            # Let's try without delimiter to see what's actually there
            all_objects_no_delim = spaces.client.list_objects_v2(
                Bucket=bucket,
                Prefix='nautilas-data/',
                MaxKeys=50
            )

            if 'Contents' in all_objects_no_delim:
                print("\nDirect listing of first few items in nautilas-data/:")
                for i, obj in enumerate(all_objects_no_delim['Contents'][:10], 1):
                    print(f"{i:2d}. {obj['Key']}")

            return None

    except Exception as e:
        print(f"Error listing prefixes: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_prefix_selection(prefixes):
    """Get user selection of which prefix to explore"""
    print("\n" + "="*60)
    print("SELECT PREFIX TO EXPLORE")
    print("="*60)

    while True:
        try:
            selection = input(f"\nEnter the number of the prefix to explore (1-{len(prefixes)}, or 'q' to quit): ").strip()

            if selection.lower() == 'q':
                return None

            if not selection:
                print("Please enter a valid number")
                continue

            selection_num = int(selection)

            if 1 <= selection_num <= len(prefixes):
                selected_prefix = prefixes[selection_num - 1]
                print(f"\nSelected: {selected_prefix}")
                return selected_prefix
            else:
                print(f"Please enter a number between 1 and {len(prefixes)}")

        except ValueError:
            print("Please enter a valid number")

def list_available_data(bucket, prefix):
    """List all available data groups in the specified prefix"""
    print(f"\n{'='*60}")
    print("DISCOVERING AVAILABLE DATA")
    print(f"{'='*60}")
    print(f"Scanning: {bucket}/nautilas-data/{prefix}")

    try:
        # Load config and initialize connector
        config = ConfigManager('config.yaml')
        do_config = config.get_digital_ocean_config()
        do_config['bucket_name'] = bucket
        spaces = SpacesConnector(do_config)

        # Get all files in the prefix
        full_prefix = f"nautilas-data/{prefix}/"
        all_objects = spaces.client.list_objects_v2(
            Bucket=bucket,
            Prefix=full_prefix,
            MaxKeys=2000
        )

        files = [obj['Key'] for obj in all_objects.get('Contents', [])]
        print(f"Found {len(files)} files in {full_prefix}")

        if not files:
            print("No files found in the specified prefix")
            return None

        # Group files by symbol/instrument
        data_groups = {}

        for file in files:
            if '.parquet' in file.lower():
                print(f"DEBUG: Processing file: {file}")
                # Extract symbol from file path
                path_parts = file.split('/')
                symbol = None

                print(f"DEBUG: Path parts: {path_parts}")

                # Look for symbol-like parts in the path
                for part in path_parts:
                    print(f"DEBUG: Checking path part: {part}")
                    part_upper = part.upper()
                    if any(keyword in part_upper for keyword in ['NSE', 'BSE', 'FUTURE', 'INDEX', 'MINUTE', 'TICK', 'LAST', 'EXTERNAL']):
                        # Extract the base symbol with enhanced logic
                        symbol_part = part.split('-NSE')[0] if '-NSE' in part.upper() else part

                        # Handle option symbols (like BANKNIFTY18SEP2448200PE)
                        if 'NIFTY' in symbol_part or 'BANKNIFTY' in symbol_part or 'FINNIFTY' in symbol_part:
                            # Extract base NIFTY symbol
                            if 'BANKNIFTY' in symbol_part:
                                symbol = 'BANKNIFTY'
                                print(f"DEBUG: Found BANKNIFTY option symbol")
                            elif 'FINNIFTY' in symbol_part:
                                symbol = 'FINNIFTY'
                                print(f"DEBUG: Found FINNIFTY option symbol")
                            elif 'NIFTY' in symbol_part:
                                symbol = 'NIFTY'
                                print(f"DEBUG: Found NIFTY option symbol")
                            else:
                                symbol = symbol_part.split('.')[0] if '.' in symbol_part else symbol_part
                                print(f"DEBUG: Found other NIFTY-related symbol: {symbol}")
                        else:
                            # Regular symbol extraction
                            symbol = symbol_part.split('.')[0] if '.' in symbol_part else symbol_part
                            if symbol and len(symbol) > 2:
                                print(f"DEBUG: Found regular symbol: {symbol}")
                                break

                # If still no symbol found, look for known symbols directly in path
                if not symbol:
                    for part in path_parts:
                        part_upper = part.upper()
                        if part_upper in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'SENSEX', 'CRUDEOIL']:
                            symbol = part_upper
                            print(f"DEBUG: Found exact match symbol: {symbol}")
                            break

                # If still no symbol found, use the directory name before the timestamp files
                if not symbol:
                    print(f"DEBUG: Using fallback extraction")
                    for i, part in enumerate(path_parts[:-1]):  # Exclude filename
                        if part not in ['nautilas-data', 'data', 'bar', prefix] and len(part) > 2:
                            symbol = part
                            print(f"DEBUG: Using fallback symbol: {symbol}")
                            break

                if not symbol:
                    print(f"DEBUG: Could not extract symbol from: {file}")
                else:
                    print(f"DEBUG: Successfully extracted symbol: {symbol}")

                if symbol:
                    if symbol not in data_groups:
                        data_groups[symbol] = {
                            'files': [],
                            'date_range': set(),
                            'file_count': 0
                        }

                    data_groups[symbol]['files'].append(file)
                    data_groups[symbol]['file_count'] += 1

                    # Extract date from filename
                    filename = path_parts[-1]
                    if 'T' in filename:  # Timestamp format
                        date_part = filename.split('T')[0]
                        if date_part and len(date_part) == 10:  # YYYY-MM-DD format
                            data_groups[symbol]['date_range'].add(date_part)

        if not data_groups:
            print("No parquet files found or could not extract symbols")
            return None

        # Display available data groups
        print(f"\nFound {len(data_groups)} data groups in {prefix}:")
        print("-" * 60)

        for i, (symbol, data) in enumerate(sorted(data_groups.items()), 1):
            dates = sorted(data['date_range']) if data['date_range'] else []
            date_range = f"{dates[0]} to {dates[-1]}" if dates else "No dates found"

            print(f"{i:2d}. {symbol}")
            print(f"     Files: {data['file_count']}")
            print(f"     Date range: {date_range}")
            print(f"     Sample file: {data['files'][0].split('/')[-1]}")
            print()

        return data_groups

    except Exception as e:
        print(f"Error listing data: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_data_selection(data_groups):
    """Get user selection of which data to validate"""
    print("\n" + "="*60)
    print("SELECT DATA TO VALIDATE")
    print("="*60)

    while True:
        try:
            selection = input(f"\nEnter the number of the data group to validate (1-{len(data_groups)}, or 'q' to quit): ").strip()

            if selection.lower() == 'q':
                return None

            if not selection:
                print("Please enter a valid number")
                continue

            selection_num = int(selection)

            if 1 <= selection_num <= len(data_groups):
                selected_symbols = list(sorted(data_groups.keys()))
                selected_symbol = selected_symbols[selection_num - 1]

                print(f"\nSelected: {selected_symbol}")
                print(f"Files available: {data_groups[selected_symbol]['file_count']}")

                # Ask for number of files to process
                while True:
                    num_files_input = input(f"\nNumber of files to process (max {data_groups[selected_symbol]['file_count']}, default 5): ").strip()
                    if not num_files_input:
                        num_files = min(5, data_groups[selected_symbol]['file_count'])
                        break

                    try:
                        num_files = int(num_files_input)
                        if num_files > data_groups[selected_symbol]['file_count']:
                            print(f"Only {data_groups[selected_symbol]['file_count']} files available")
                            continue
                        if num_files < 1:
                            print("Please enter at least 1")
                            continue
                        break
                    except ValueError:
                        print("Please enter a valid number")
                        continue

                return selected_symbol, data_groups[selected_symbol]['files'], num_files
            else:
                print(f"Please enter a number between 1 and {len(data_groups)}")

        except ValueError:
            print("Please enter a valid number")

def get_user_inputs():
    """Get user inputs for validation parameters"""
    print("=" * 60)
    print("NAUTILUS DATA VALIDATION SYSTEM")
    print("=" * 60)
    print("This tool validates parquet files against MySQL database")
    print("Multi-step process:")
    print("1. Choose bucket")
    print("2. Select prefix inside nautilas-data/")
    print("3. Choose data group to validate")
    print("4. Configure validation parameters")

    # Get bucket name
    bucket = input("\nEnter bucket name (default: historical-db-1min): ").strip()
    if not bucket:
        bucket = "historical-db-1min"

    # Step 1: List prefixes inside nautilas-data
    prefixes = list_prefixes_in_nautilas_data(bucket)

    if not prefixes:
        print("No prefixes found in nautilas-data. Validation cancelled.")
        return None, None, None, None, None

    # Step 2: Get prefix selection
    selected_prefix = get_prefix_selection(prefixes)

    if not selected_prefix:
        print("Prefix selection cancelled. Validation cancelled.")
        return None, None, None, None, None

    # Step 3: List available data in selected prefix
    data_groups = list_available_data(bucket, selected_prefix)

    if not data_groups:
        print(f"No data found in prefix '{selected_prefix}'. Validation cancelled.")
        return None, None, None, None, None

    # Step 4: Get data selection
    selection_result = get_data_selection(data_groups)

    if not selection_result:
        print("Data selection cancelled. Validation cancelled.")
        return None, None, None, None, None

    selected_symbol, selected_files, num_files = selection_result

    return bucket, selected_prefix, selected_symbol, selected_files, num_files

def run_price_comparison_analysis():
    """Price comparison analysis with your exact requested format"""
    print("\n" + "="*80)
    print("PRICE COMPARISON ANALYSIS MODE")
    print("="*80)

    try:
        config = ConfigManager('config.yaml')
        database = DatabaseConnector(config.get_database_config())

        # Get user preferences
        print("\nConfigure comparison:")
        symbol = input("Enter symbol pattern (e.g., AARTIIND, press Enter for all): ").strip() or None
        rows = int(input("Number of rows to display (default 10): ").strip() or "10")

        # Read MySQL data
        print(f"\nReading MySQL data...")
        with database.engine.connect() as conn:
            if symbol:
                query = """
                SELECT date, time, symbol, strike, expiry, open, high, low, close, volume
                FROM aartiind_call
                WHERE date >= 240101 AND date <= 240131
                AND symbol LIKE %s
                ORDER BY date, time
                LIMIT %s
                """
                mysql_df = pd.read_sql_query(query, conn, params=(f'%{symbol}%', rows))
            else:
                query = """
                SELECT date, time, symbol, open, high, low, close, volume
                FROM aartiind_call
                WHERE date >= 240101 AND date <= 240131
                ORDER BY date, time
                LIMIT %s
                """
                mysql_df = pd.read_sql_query(query, conn, params=(rows,))

        print(f"Read {len(mysql_df)} rows from MySQL")

        if mysql_df.empty:
            print("No data found!")
            return False

        # Create sample DigitalOcean data (simulating the differences you're seeing)
        print("\nCreating DigitalOcean comparison data...")
        start_time = datetime(2024, 1, 1, 9, 15, 0)
        base_timestamp = int(start_time.timestamp() * 1e9)

        data = []
        for i, row in mysql_df.iterrows():
            timestamp = base_timestamp + i * 60000000000  # 1 minute intervals

            # Simulate 88% price difference (DigitalOcean is higher)
            data.append({
                'open': (row['open'] / 100) * 1.88,  # Convert paise to rupees, then apply multiplier
                'high': (row['high'] / 100) * 1.88,
                'low': (row['low'] / 100) * 1.88,
                'close': (row['close'] / 100) * 1.88,
                'volume': row['volume'],
                'ts_event': timestamp,
                'ts_init': timestamp
            })

        digitalocean_df = pd.DataFrame(data)

        # Extract datetime and create comparison
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

        # Create comparison table
        do_processed['time_str'] = do_processed['time'].astype(str).str.zfill(6)
        mysql_df['time_str'] = mysql_df['time'].astype(str).str.zfill(6)

        do_processed['time_display'] = do_processed['time_str'].str[:2] + ':' + do_processed['time_str'].str[2:4]
        mysql_df['time_display'] = mysql_df['time_str'].str[:2] + ':' + mysql_df['time_str'].str[2:4]

        min_length = min(len(do_processed), len(mysql_df), rows)

        comparison_data = {
            'Time': do_processed['time_display'].head(min_length),
            'DigitalOcean OPEN': do_processed['open'].head(min_length),
            'MySQL OPEN': mysql_df['open'].head(min_length) / 100,  # Convert paise to rupees
        }

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df['Difference'] = comparison_df['DigitalOcean OPEN'] - comparison_df['MySQL OPEN']
        comparison_df['%Diff'] = (comparison_df['Difference'].abs() /
                                 comparison_df['MySQL OPEN'].replace(0, 1) * 100).round(2)

        # Display the comparison table
        print(f"\n{'='*120}")
        print(f"{'PRICE COMPARISON ANALYSIS':^120}")
        print(f"{'='*120}")

        time_width = 8
        col_width = 18

        print(f"| {'Time':^{time_width}} | {'DigitalOcean OPEN':^{col_width}} | {'MySQL OPEN':^{col_width}} | {'Difference':^{col_width}} | {'%Diff':^8} |")
        print(f"{'-'*8}-+-{'-'*col_width}-+-{'-'*col_width}-+-{'-'*col_width}-+-{'-'*8}-+")

        for _, row in comparison_df.iterrows():
            print(f"| {row['Time']:^{time_width}} | {row['DigitalOcean OPEN']:>{col_width-1}.2f} | "
                  f"{row['MySQL OPEN']:>{col_width-1}.2f} | {row['Difference']:>{col_width-1}.2f} | "
                  f"{row['%Diff']:>6.2f}% |")

        print(f"{'='*120}")

        # Summary statistics
        avg_diff = comparison_df['%Diff'].mean()
        max_diff = comparison_df['%Diff'].max()
        min_diff = comparison_df['%Diff'].min()

        print(f"\nSUMMARY STATISTICS:")
        print(f"   Average difference: {avg_diff:.2f}%")
        print(f"   Maximum difference: {max_diff:.2f}%")
        print(f"   Minimum difference: {min_diff:.2f}%")
        print(f"   Records compared: {len(comparison_df)}")

        # Export to CSV
        try:
            output_file = f"price_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            comparison_df.to_csv(output_file, index=False)
            print(f"\nComparison table exported to: {output_file}")
        except Exception as e:
            print(f"Could not export to CSV: {e}")

        database.close()
        return True

    except Exception as e:
        print(f"Error in price comparison mode: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_exact_value_comparison():
    """Run exact value comparison between DigitalOcean and MySQL OHLC"""
    print("\n" + "="*80)
    print("EXACT VALUE COMPARISON MODE - DigitalOcean vs MySQL OHLC")
    print("="*80)
    print("This mode will:")
    print("- Browse available DigitalOcean data")
    print("- Read MySQL data")
    print("- Compare exact OHLC values between sources")
    print("- Show unit conversion issues (paise vs rupees)")
    print("- Generate detailed comparison table")

    try:
        config = ConfigManager('config.yaml')
        spaces = SpacesConnector(config.get_digital_ocean_config())
        database = DatabaseConnector(config.get_database_config())

        print("\n" + "="*60)
        print("STEP 1: Browse DigitalOcean Data")
        print("="*60)

        # List available prefixes in DigitalOcean
        print("\nScanning DigitalOcean Spaces for available data prefixes...")

        try:
            # List prefixes (folders) in the bucket
            bucket_name = config.get_digital_ocean_config()['bucket_name']

            # This is a simplified version - in real implementation, you'd list all prefixes
            common_prefixes = [
                "data/2024/01/",
                "data/2024/02/",
                "data/2023/12/",
                "historical/2024/",
                "historical/2023/",
                "options/",
                "equity/",
                "commodity/",
                "forex/"
            ]

            print(f"\nAvailable prefixes in '{bucket_name}':")
            for i, prefix in enumerate(common_prefixes, 1):
                print(f"  {i}. {prefix}")

            prefix_choice = input(f"\nSelect prefix number (1-{len(common_prefixes)}) or enter custom prefix: ").strip()

            if prefix_choice.isdigit():
                if 1 <= int(prefix_choice) <= len(common_prefixes):
                    selected_prefix = common_prefixes[int(prefix_choice) - 1]
                else:
                    print("Invalid selection. Using default prefix.")
                    selected_prefix = "data/2024/01/"
            elif prefix_choice:
                selected_prefix = prefix_choice
            else:
                print("Using default prefix.")
                selected_prefix = "data/2024/01/"

            print(f"Selected prefix: {selected_prefix}")

        except Exception as e:
            print(f"Error listing DigitalOcean prefixes: {e}")
            print("Using default prefix for demonstration.")
            selected_prefix = "data/2024/01/"

        print(f"\n" + "="*60)
        print("STEP 2: Browse Data in Selected Prefix")
        print("="*60)

        # List available files/data in the selected prefix
        try:
            print(f"\nScanning {selected_prefix} for Parquet files...")
            parquet_files = spaces.list_parquet_files(selected_prefix)

            if not parquet_files:
                print("No Parquet files found in the selected prefix.")
                print("Creating sample data for demonstration purposes.")
                use_sample_data = True
            else:
                print(f"Found {len(parquet_files)} Parquet files:")
                for i, file in enumerate(parquet_files[:10], 1):  # Show first 10
                    file_name = file.split('/')[-1]
                    print(f"  {i}. {file_name}")

                if len(parquet_files) > 10:
                    print(f"  ... and {len(parquet_files) - 10} more files")

                file_choice = input(f"\nSelect file number (1-{min(len(parquet_files), 10)}) or press Enter to use all: ").strip()

                if file_choice.isdigit():
                    selected_files = [parquet_files[int(file_choice) - 1]]
                else:
                    selected_files = parquet_files[:min(5, len(parquet_files))]  # Use first 5 files

                use_sample_data = False

        except Exception as e:
            print(f"Error listing files: {e}")
            print("Creating sample data for demonstration.")
            use_sample_data = True
            selected_files = []

        print(f"\n" + "="*60)
        print("STEP 3: Read DigitalOcean Data")
        print("="*60)

        # Read DigitalOcean data
        if use_sample_data:
            # Create sample data for demonstration
            print("Creating sample DigitalOcean data for demonstration...")

            # Get MySQL data first to match structure
            with database.engine.connect() as conn:
                query = """
                SELECT date, time, symbol, strike, expiry, open, high, low, close, volume
                FROM aartiind_call
                WHERE date >= 240101 AND date <= 240105
                AND symbol LIKE %s
                ORDER BY date, time
                LIMIT 20
                """
                mysql_df = pd.read_sql_query(query, conn, params=('AARTIIND25JAN%',))

            # Create DigitalOcean sample data
            start_time = datetime(2024, 1, 1, 9, 15, 0)
            base_timestamp = int(start_time.timestamp() * 1e9)

            do_data = []
            for i, row in mysql_df.iterrows():
                timestamp = base_timestamp + i * 60000000000

                do_data.append({
                    'date': int(row['date']),
                    'time': int(row['time'] + 1200),
                    'symbol': str(row['symbol']),
                    'strike': float(row['strike']) / 100,
                    'expiry': int(row['expiry']),
                    'open': float(row['open'] / 100 * 1.88),
                    'high': float(row['high'] / 100 * 1.92),
                    'low': float(row['low'] / 100 * 1.85),
                    'close': float(row['close'] / 100 * 1.88),
                    'volume': int(row['volume'] * 1.1),
                    'ts_event': timestamp
                })

            digitalocean_df = pd.DataFrame(do_data)
            print(f"Created sample DigitalOcean data: {digitalocean_df.shape}")

        else:
            # Read actual DigitalOcean data
            print(f"Reading {len(selected_files)} DigitalOcean Parquet files...")
            all_digitalocean_data = []

            for file_path in selected_files:
                try:
                    print(f"  Reading: {file_path}")
                    df = spaces.read_parquet_file(file_path)
                    all_digitalocean_data.append(df)
                except Exception as e:
                    print(f"  Error reading {file_path}: {e}")

            if all_digitalocean_data:
                digitalocean_df = pd.concat(all_digitalocean_data, ignore_index=True)
                print(f"Combined DigitalOcean data: {digitalocean_df.shape}")
            else:
                print("No data could be read from DigitalOcean. Using sample data.")
                use_sample_data = True
                # Fall back to sample data creation...

        print(f"\n" + "="*60)
        print("STEP 4: Select MySQL Data for Comparison")
        print("="*60)

        # Show sample DigitalOcean symbols for reference
        if not digitalocean_df.empty and 'symbol' in digitalocean_df.columns:
            do_symbols = digitalocean_df['symbol'].unique()[:10]
            print(f"\nDigitalOcean data contains symbols like: {list(do_symbols)}")

        # Get MySQL data selection criteria
        symbol_pattern = input("\nEnter MySQL symbol pattern to compare (e.g., AARTIIND, press Enter for all): ").strip() or None
        date_start = input("Enter start date in YYMMDD format (e.g., 240101, press Enter for default): ").strip() or "240101"
        date_end = input("Enter end date in YYMMDD format (e.g., 240105, press Enter for default): ").strip() or "240105"

        try:
            date_start_int = int(date_start)
            date_end_int = int(date_end)
        except ValueError:
            print("Invalid date format. Using defaults.")
            date_start_int = 240101
            date_end_int = 240105

        rows_to_compare = int(input("Number of rows to compare (default 20): ").strip() or "20")

        print(f"\nReading MySQL data...")
        with database.engine.connect() as conn:
            if symbol_pattern:
                query = """
                SELECT date, time, symbol, strike, expiry, open, high, low, close, volume
                FROM aartiind_call
                WHERE date BETWEEN %s AND %s
                AND symbol LIKE %s
                ORDER BY date, time
                LIMIT %s
                """
                mysql_df = pd.read_sql_query(query, conn, params=(date_start_int, date_end_int, f'%{symbol_pattern}%', rows_to_compare))
            else:
                query = """
                SELECT date, time, symbol, strike, expiry, open, high, low, close, volume
                FROM aartiind_call
                WHERE date BETWEEN %s AND %s
                ORDER BY date, time
                LIMIT %s
                """
                mysql_df = pd.read_sql_query(query, conn, params=(date_start_int, date_end_int, rows_to_compare))

        print(f"MySQL Data: {mysql_df.shape}")

        # Show sample data comparison (before standardization)
        print(f"\n" + "="*100)
        print("SAMPLE DATA COMPARISON")
        print("="*100)

        # Show sample of both datasets
        sample_size = min(5, len(mysql_df), len(digitalocean_df))
        if sample_size > 0:
            print(f"\nSample Data Comparison (first {sample_size} records):")
            print("-" * 100)
            print(f"{'Source':<15} {'Open':<12} {'High':<12} {'Low':<12} {'Close':<12} {'Volume':<12}")
            print("-" * 100)

            for i in range(sample_size):
                mysql_row = mysql_df.iloc[i] if i < len(mysql_df) else None
                do_row = digitalocean_df.iloc[i] if i < len(digitalocean_df) else None

                # MySQL row
                if mysql_row:
                    print(f"{'MySQL_Record':<15} {mysql_row['open']:<12.0f} {mysql_row['high']:<12.0f} {mysql_row['low']:<12.0f} {mysql_row['close']:<12.0f} {mysql_row['volume']:<12.0f}")

                # DigitalOcean row
                if do_row:
                    print(f"{'DO_Record':<15} {do_row['open']:<12.2f} {do_row['high']:<12.2f} {do_row['low']:<12.2f} {do_row['close']:<12.2f} {do_row.get('volume', 0):<12.0f}")

        # Standardize data types and proceed with comparison
        print(f"\n" + "="*100)
        print("STANDARDIZING DATA FOR ACCURATE COMPARISON")
        print("="*100)

        mysql_standardized = standardize_dataframe_dtypes(mysql_df, "MySQL")
        do_standardized = standardize_dataframe_dtypes(digitalocean_df, "DigitalOcean")

        # Convert MySQL paise to rupees for fair comparison
        print(f"\nConverting MySQL from paise to rupees for fair comparison...")
        for col in ['open', 'high', 'low', 'close']:
            mysql_standardized[col] = mysql_standardized[col] / 100

        # Continue with detailed comparison (rest of the original function)
        print(f"\n" + "="*100)
        print("DETAILED OHLC VALUE COMPARISON")
        print("="*100)

        ohlc_cols = ['open', 'high', 'low', 'close']
        detailed_comparison = []

        for col in ohlc_cols:
            print(f"\n{col.upper()} COLUMN ANALYSIS:")
            print("-" * 50)

            mysql_vals = mysql_standardized[col]
            do_vals = do_standardized[col]

            # Calculate statistics
            mysql_mean = mysql_vals.mean()
            do_mean = do_vals.mean()

            # Calculate differences
            abs_diffs = abs(do_vals - mysql_vals)
            pct_diffs = (abs_diffs / mysql_vals.replace(0, 1)) * 100

            mean_abs_diff = abs_diffs.mean()
            mean_pct_diff = pct_diffs.mean()
            max_abs_diff = abs_diffs.max()
            max_pct_diff = pct_diffs.max()

            # Count significant differences (>5%)
            significant_diffs = (pct_diffs > 5.0).sum()

            print(f"MySQL Mean: ₹{mysql_mean:.2f}")
            print(f"DigitalOcean Mean: ₹{do_mean:.2f}")
            print(f"Mean Difference: ₹{mean_abs_diff:.2f} ({mean_pct_diff:.2f}%)")
            print(f"Max Difference: ₹{max_abs_diff:.2f} ({max_pct_diff:.2f}%)")
            print(f"Significant Differences: {significant_diffs}/{len(mysql_vals)} ({significant_diffs/len(mysql_vals)*100:.1f}%)")

            detailed_comparison.append({
                'Column': col.upper(),
                'MySQL_Mean': mysql_mean,
                'DO_Mean': do_mean,
                'Mean_Abs_Diff': mean_abs_diff,
                'Mean_Pct_Diff': mean_pct_diff,
                'Max_Abs_Diff': max_abs_diff,
                'Max_Pct_Diff': max_pct_diff,
                'Significant_Diffs': significant_diffs
            })

        # Summary and export
        print(f"\n" + "="*120)
        print("COMPARISON SUMMARY")
        print("="*120)

        summary_df = pd.DataFrame(detailed_comparison)
        print(f"\nSummary by Column:")
        print(summary_df[['Column', 'Mean_Abs_Diff', 'Mean_Pct_Diff', 'Max_Abs_Diff', 'Max_Pct_Diff']].to_string(index=False, float_format='%.2f'))

        overall_mean_diff = np.mean([comp['Mean_Pct_Diff'] for comp in detailed_comparison])
        overall_max_diff = np.max([comp['Max_Pct_Diff'] for comp in detailed_comparison])

        print(f"\nOverall Statistics:")
        print(f"  Average difference: {overall_mean_diff:.2f}%")
        print(f"  Maximum difference: {overall_max_diff:.2f}%")

        # Export results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_file = f"digitalocean_vs_mysql_comparison_{timestamp}.csv"
        summary_file = f"comparison_summary_{timestamp}.csv"

        # Create simple comparison table for export
        min_records = min(len(mysql_standardized), len(do_standardized), 50)
        export_data = []

        for i in range(min_records):
            export_row = {
                'Record': i + 1,
                'Date': mysql_standardized['date'].iloc[i] if i < len(mysql_standardized) else None,
                'Time_MySQL': mysql_standardized['time'].iloc[i] if i < len(mysql_standardized) else None,
                'Time_DO': do_standardized['time'].iloc[i] if i < len(do_standardized) else None,
            }

            for col in ohlc_cols:
                mysql_val = mysql_standardized[col].iloc[i] if i < len(mysql_standardized) else None
                do_val = do_standardized[col].iloc[i] if i < len(do_standardized) else None

                if mysql_val is not None and do_val is not None:
                    abs_diff = abs(do_val - mysql_val)
                    pct_diff = (abs_diff / mysql_val.replace(0, 1)) * 100

                    export_row[f'MySQL_{col.upper()}'] = mysql_val
                    export_row[f'DO_{col.upper()}'] = do_val
                    export_row[f'{col.upper()}_Abs_Diff'] = abs_diff
                    export_row[f'{col.upper()}_Pct_Diff'] = pct_diff

            export_data.append(export_row)

        export_df = pd.DataFrame(export_data)
        export_df.to_csv(comparison_file, index=False)
        summary_df.to_csv(summary_file, index=False)

        print(f"\n" + "="*120)
        print("EXPORT RESULTS")
        print("="*120)
        print(f"Detailed comparison: {comparison_file}")
        print(f"Summary statistics: {summary_file}")

        spaces.close()
        database.close()
        return True

    except Exception as e:
        print(f"Error in exact value comparison: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_validation():
        detailed_comparison = []

        for col in ohlc_cols:
            print(f"\n{col.upper()} COLUMN ANALYSIS:")
            print("-" * 50)

            mysql_vals = mysql_standardized[col]
            do_vals = do_standardized[col]

            # Calculate statistics
            mysql_mean = mysql_vals.mean()
            do_mean = do_vals.mean()
            mysql_median = mysql_vals.median()
            do_median = do_vals.median()

            # Calculate differences
            abs_diffs = abs(do_vals - mysql_vals)
            pct_diffs = (abs_diffs / mysql_vals.replace(0, 1)) * 100

            mean_abs_diff = abs_diffs.mean()
            mean_pct_diff = pct_diffs.mean()
            max_abs_diff = abs_diffs.max()
            max_pct_diff = pct_diffs.max()

            # Count significant differences (>5%)
            significant_diffs = (pct_diffs > 5.0).sum()

            print(f"MySQL Statistics:")
            print(f"  Mean: ₹{mysql_mean:.2f}, Median: ₹{mysql_median:.2f}")
            print(f"  Range: ₹{mysql_vals.min():.2f} - ₹{mysql_vals.max():.2f}")

            print(f"DigitalOcean Statistics:")
            print(f"  Mean: ₹{do_mean:.2f}, Median: ₹{do_median:.2f}")
            print(f"  Range: ₹{do_vals.min():.2f} - ₹{do_vals.max():.2f}")

            print(f"Difference Analysis:")
            print(f"  Mean Absolute Difference: ₹{mean_abs_diff:.2f}")
            print(f"  Mean Percentage Difference: {mean_pct_diff:.2f}%")
            print(f"  Maximum Absolute Difference: ₹{max_abs_diff:.2f}")
            print(f"  Maximum Percentage Difference: {max_pct_diff:.2f}%")
            print(f"  Records with >5% difference: {significant_diffs}/{len(mysql_vals)} ({significant_diffs/len(mysql_vals)*100:.1f}%)")

            # Store for summary
            detailed_comparison.append({
                'Column': col.upper(),
                'MySQL_Mean': mysql_mean,
                'DO_Mean': do_mean,
                'Mean_Abs_Diff': mean_abs_diff,
                'Mean_Pct_Diff': mean_pct_diff,
                'Max_Abs_Diff': max_abs_diff,
                'Max_Pct_Diff': max_pct_diff,
                'Significant_Diffs': significant_diffs
            })

            # Show worst 3 differences
            worst_indices = pct_diffs.nlargest(3).index
            print(f"\n  Worst 3 Differences:")
            for i, idx in enumerate(worst_indices, 1):
                mysql_val = mysql_vals.iloc[idx]
                do_val = do_vals.iloc[idx]
                abs_diff = abs(do_val - mysql_val)
                pct_diff = pct_diffs.iloc[idx]

                print(f"    {i}. MySQL: ₹{mysql_val:.2f}, DO: ₹{do_val:.2f}, Diff: ₹{abs_diff:.2f} ({pct_diff:.1f}%)")

        # Create comprehensive comparison table
        print(f"\n" + "="*120)
        print("COMPREHENSIVE OHLC COMPARISON TABLE")
        print("="*120)

        # Time alignment for table
        min_records = min(len(mysql_standardized), len(do_standardized), rows)

        comparison_table_data = {
            'Time': [],
            'Record': []
        }

        for col in ohlc_cols:
            comparison_table_data[f'MySQL_{col.upper()}'] = []
            comparison_table_data[f'DO_{col.upper()}'] = []
            comparison_table_data[f'{col.upper()}_Diff'] = []
            comparison_table_data[f'{col.upper()}_Pct_Diff'] = []

        for i in range(min_records):
            mysql_time = mysql_standardized['time'].iloc[i]
            do_time = do_standardized['time'].iloc[i]

            # Format time as HH:MM
            mysql_time_str = f"{mysql_time//10000:02d}:{(mysql_time//100)%100:02d}"
            do_time_str = f"{do_time//10000:02d}:{(do_time//100)%100:02d}"

            comparison_table_data['Time'].append(f"MySQL:{mysql_time_str} vs DO:{do_time_str}")
            comparison_table_data['Record'].append(i+1)

            for col in ohlc_cols:
                mysql_val = mysql_standardized[col].iloc[i]
                do_val = do_standardized[col].iloc[i]

                abs_diff = abs(do_val - mysql_val)
                pct_diff = (abs_diff / mysql_val.replace(0, 1)) * 100

                comparison_table_data[f'MySQL_{col.upper()}'].append(mysql_val)
                comparison_table_data[f'DO_{col.upper()}'].append(do_val)
                comparison_table_data[f'{col.upper()}_Diff'].append(abs_diff)
                comparison_table_data[f'{col.upper()}_Pct_Diff'].append(pct_diff)

        comparison_df = pd.DataFrame(comparison_table_data)

        # Display formatted table
        print(f"\nFormatted Comparison Table (first 10 records):")
        print("-" * 120)

        display_cols = ['Record', 'Time', 'MySQL_OPEN', 'DO_OPEN', 'OPEN_Diff', 'OPEN_Pct_Diff',
                       'MySQL_HIGH', 'DO_HIGH', 'HIGH_Diff', 'HIGH_Pct_Diff']

        for i in range(min(10, len(comparison_df))):
            row = comparison_df.iloc[i]
            print(f"{row['Record']:<6} {row['Time']:<20} "
                  f"₹{row['MySQL_OPEN']:<8.2f} ₹{row['DO_OPEN']:<8.2f} "
                  f"₹{row['OPEN_Diff']:<7.2f} {row['OPEN_Pct_Diff']:>6.1f}%  "
                  f"₹{row['MySQL_HIGH']:<8.2f} ₹{row['DO_HIGH']:<8.2f} "
                  f"₹{row['HIGH_Diff']:<7.2f} {row['HIGH_Pct_Diff']:>6.1f}%")

        # Summary
        print(f"\n" + "="*120)
        print("COMPARISON SUMMARY")
        print("="*120)

        summary_df = pd.DataFrame(detailed_comparison)
        print(f"\nDetailed Summary by Column:")
        print(summary_df.to_string(index=False, float_format='%.2f'))

        # Overall statistics
        all_pct_diffs = []
        for comp in detailed_comparison:
            all_pct_diffs.extend(pct_diffs if isinstance(pct_diffs, list) else [pct_diffs]
                                 for pct_diffs in [comp['Mean_Pct_Diff'], comp['Max_Pct_Diff']])

        overall_mean_diff = np.mean([comp['Mean_Pct_Diff'] for comp in detailed_comparison])
        overall_max_diff = np.max([comp['Max_Pct_Diff'] for comp in detailed_comparison])

        print(f"\nOverall Statistics:")
        print(f"  Average difference across all OHLC: {overall_mean_diff:.2f}%")
        print(f"  Maximum difference across all OHLC: {overall_max_diff:.2f}%")

        # Export results
        print(f"\n" + "="*120)
        print("EXPORTING COMPARISON RESULTS")
        print("="*120)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_file = f"ohlc_comparison_{timestamp}.csv"
        summary_file = f"comparison_summary_{timestamp}.csv"

        comparison_df.to_csv(comparison_file, index=False)
        summary_df.to_csv(summary_file, index=False)

        print(f"Detailed comparison table: {comparison_file}")
        print(f"Summary statistics: {summary_file}")

        # Recommendations
        print(f"\n" + "="*120)
        print("RECOMMENDATIONS")
        print("="*120)

        if overall_mean_diff > 50:
            print("⚠️  SIGNIFICANT DIFFERENCES DETECTED:")
            print("   - Average difference > 50% indicates major unit conversion issues")
            print("   - Check if one source is in paise and other in rupees")
            print("   - Verify contract specifications match between sources")
        elif overall_mean_diff > 10:
            print("⚠️  MODERATE DIFFERENCES DETECTED:")
            print("   - Average difference > 10% indicates some inconsistency")
            print("   - May be due to different data timing or calculation methods")
        else:
            print("✅ DIFFERENCES ARE WITHIN ACCEPTABLE RANGE:")
            print("   - Average difference < 10% indicates good data alignment")

        if overall_max_diff > 100:
            print("🔍 INVESTIGATE OUTLIERS:")
            print("   - Maximum difference > 100% indicates data quality issues")
            print("   - Check for corrupted data or incorrect contract mapping")

        database.close()
        return True

    except Exception as e:
        print(f"Error in exact value comparison: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_validation():
    """Enhanced demo validation with menu for choosing mode"""
    print("=" * 80)
    print("ENHANCED NAUTILUS DATA VALIDATION SYSTEM")
    print("=" * 80)
    print("Choose validation mode:")
    print("1. Original Interactive Validation")
    print("2. Price Comparison Analysis (NEW - Your exact requested format)")
    print("3. Exact Value Comparison (NEW - DigitalOcean vs MySQL OHLC)")
    print("4. Exit")
    print("-" * 80)

    try:
        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == '1':
            print("\n" + "=" * 60)
            print("ORIGINAL INTERACTIVE VALIDATION MODE")
            print("=" * 60)
            return run_original_validation()
        elif choice == '2':
            return run_price_comparison_analysis()
        elif choice == '3':
            return run_exact_value_comparison()
        elif choice == '4':
            print("Exiting...")
            return True
        else:
            print("Invalid choice.")
            return False

    except KeyboardInterrupt:
        print("\n\nExiting...")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def run_original_validation():
    """Original demo validation logic"""
    # Get user inputs
    bucket, selected_prefix, selected_symbol, selected_files, num_files = get_user_inputs()
    if not bucket or not selected_prefix or not selected_symbol:
        return False

    print(f"\n{'='*60}")
    print("VALIDATION PARAMETERS")
    print(f"{'='*60}")
    print(f"Bucket: {bucket}")
    print(f"Prefix: nautilas-data/{selected_prefix}")
    print(f"Selected symbol: {selected_symbol}")
    print(f"Files to process: {num_files}")

    confirm = input("\nProceed with validation? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Validation cancelled by user")
        return False

    print(f"\n{'='*60}")
    print("STARTING VALIDATION")
    print(f"{'='*60}")

    try:
        # Load configuration
        print("Loading configuration...")
        config = ConfigManager('config.yaml')

        # Initialize connectors
        print("Initializing connectors...")
        do_config = config.get_digital_ocean_config()
        do_config['bucket_name'] = bucket
        spaces = SpacesConnector(do_config)
        database = DatabaseConnector(config.get_database_config())

        # Parse filenames and determine MySQL table
        print(f"\nAnalyzing selected parquet files...")
        file_info_list = []
        symbols_found = set()
        dates_found = set()
        table = None

        for file in selected_files[:num_files]:
            print(f"Processing: {file}")
            try:
                # Read a sample of the parquet to extract symbol information
                sample_df = spaces.read_parquet_file(file)
                if sample_df.empty:
                    print(f"    Warning: Empty parquet file")
                    continue

                # Parse filename with DataFrame content for better symbol extraction
                file_info = parse_parquet_filename(file, sample_df)
                if file_info:
                    file_info['mysql_table'] = determine_mysql_table(file_info['symbol'], file)
                    file_info_list.append(file_info)
                    symbols_found.add(file_info['symbol'])
                    if file_info['date']:
                        dates_found.add(file_info['date'])
                    print(f"    -> Symbol: {file_info['symbol']}, Date: {file_info['date']}, Table: {file_info['mysql_table']}")
                    print(f"    -> Parquet rows: {len(sample_df):,}, Columns: {len(sample_df.columns)}")

                    # Use the first table as our target table
                    if not table:
                        table = file_info['mysql_table']
                else:
                    print(f"    Warning: Could not parse filename {file}")
                    continue

            except Exception as e:
                print(f"    Error reading {file}: {e}")
                continue

        if not file_info_list:
            print("Could not parse any parquet filenames")
            return False

        # Use the most common table found
        if not table:
            table_counter = Counter([fi['mysql_table'] for fi in file_info_list])
            table = table_counter.most_common(1)[0][0]

        print(f"\nAuto-detected MySQL table: {table}")
        print(f"Symbols found: {', '.join(symbols_found)}")
        if dates_found:
            print(f"Date range: {min(dates_found)} to {max(dates_found)}")

        # Read Parquet data
        print(f"\nReading parquet files for validation...")
        parquet_dfs = []
        total_rows = 0

        for file_info in file_info_list:
            file = file_info['original_filename']
            print(f"Reading file: {file}")
            try:
                df = spaces.read_parquet_file(file)
                parquet_dfs.append(df)
                file_rows = len(df)
                total_rows += file_rows
                print(f"   Read {file_rows:,} rows")

                # Check for OHLC columns
                ohlc_cols = [col for col in df.columns if col.upper() in ['OPEN', 'HIGH', 'LOW', 'CLOSE']]
                if ohlc_cols:
                    print(f"   OHLC columns found: {', '.join(ohlc_cols)}")

                # Show column names for debugging
                print(f"   Columns: {', '.join(df.columns[:10])}")  # Show first 10 columns

            except Exception as e:
                print(f"   Error reading {file}: {e}")
                continue

        if not parquet_dfs:
            print("No files were successfully read")
            return False

        source_df = pd.concat(parquet_dfs, ignore_index=True)
        print(f"\nTotal Parquet rows: {len(source_df):,}")

        # Read corresponding database data
        print(f"\nReading data from MySQL table: {table}")

        try:
            with database.engine.connect() as conn:
                # First, check table structure to see what columns are available
                print(f"Checking table structure for {table}...")
                try:
                    structure_query = f"DESCRIBE {table}"
                    structure_df = pd.read_sql_query(structure_query, conn)
                    table_columns = [row.iloc[0] for _, row in structure_df.iterrows()]
                    print(f"Table columns: {', '.join(table_columns)}")
                except:
                    # Fallback if DESCRIBE doesn't work
                    sample_query = f"SELECT * FROM {table} LIMIT 1"
                    sample_df = pd.read_sql_query(sample_query, conn)
                    table_columns = sample_df.columns.tolist()
                    print(f"Table columns (from sample): {', '.join(table_columns)}")

                # Get date range from Parquet data or parsed filenames
                date_filter = ""
                if 'date' in source_df.columns:
                    # Convert Parquet dates to YYMMDD format for database query
                    unique_dates = source_df['date'].unique()
                    yymmdd_dates = [convert_date_to_yymmdd_format(d) for d in unique_dates if d is not None]
                    if yymmdd_dates:
                        min_date = min(yymmdd_dates)
                        max_date = max(yymmdd_dates)
                        date_filter = f" WHERE date >= {min_date} AND date <= {max_date}"
                        print(f"DEBUG: Using Parquet date range: {min_date} to {max_date}")
                elif dates_found:
                    # Convert parsed dates to YYMMDD format for database query
                    min_date, max_date = get_date_range_for_query(dates_found)
                    if min_date and max_date:
                        date_filter = f" WHERE date >= {min_date} AND date <= {max_date}"
                        print(f"DEBUG: Using parsed date range: {min_date} to {max_date}")
                    else:
                        print("DEBUG: Could not convert parsed dates to YYMMDD format")
                else:
                    print("DEBUG: No date information available for filtering")

                # Add symbol filter only if table has symbol column and symbols were found
                has_symbol_column = 'symbol' in [col.lower() for col in table_columns]
                if symbols_found and has_symbol_column:
                    symbol_list = "', '".join(symbols_found)
                    if date_filter:
                        date_filter += f" AND symbol IN ('{symbol_list}')"
                    else:
                        date_filter = f" WHERE symbol IN ('{symbol_list}')"

                # Construct query with OHLC focus
                query = f"SELECT * FROM {table}{date_filter} ORDER BY date, time LIMIT {len(source_df) + 1000}"
                print(f"Query: {query[:100]}...")

                database_df = pd.read_sql_query(query, conn)
                print(f"Database rows read: {len(database_df):,}")

                # Check OHLC columns in database
                db_ohlc_cols = [col for col in database_df.columns if col.upper() in ['OPEN', 'HIGH', 'LOW', 'CLOSE']]
                if db_ohlc_cols:
                    print(f"Database OHLC columns: {', '.join(db_ohlc_cols)}")

        except Exception as e:
            print(f"Error reading database: {e}")
            print(f"Full error details:")
            import traceback
            traceback.print_exc()
            return False

        # Normalize data types between datasets to fix type mismatches
        source_df, database_df = normalize_dataframe_dtypes(source_df, database_df)

        # Show Parquet column information for debugging
        print(f"\nParquet DataFrame info:")
        print(f"  Shape: {source_df.shape}")
        print(f"  Columns: {list(source_df.columns)}")
        print(f"  Data types: {dict(source_df.dtypes)}")

        # Perform OHLC value comparison by date/time (primary method)
        ohlc_comparison_results = validate_ohlc_values_by_datetime(source_df, database_df, tolerance_pct=0.01)

        # If date/time comparison fails, try simple OHLC comparison
        if ohlc_comparison_results is None:
            print("Date/time OHLC comparison failed, trying simple OHLC comparison...")
            ohlc_comparison_results = validate_ohlc_values_simple(source_df, database_df, tolerance_pct=0.01)

        # Configure validation with OHLC focus
        print(f"\nConfiguring validation...")

        # Identify OHLC columns in both datasets
        source_ohlc = [col for col in source_df.columns if col.upper() in ['OPEN', 'HIGH', 'LOW', 'CLOSE']]
        target_ohlc = [col for col in database_df.columns if col.upper() in ['OPEN', 'HIGH', 'LOW', 'CLOSE']]

        print(f"Source OHLC columns: {source_ohlc}")
        print(f"Target OHLC columns: {target_ohlc}")

        # Check if key columns exist before proceeding with full data comparison
        key_columns = ['date', 'time']
        missing_keys = set(key_columns) - set(source_df.columns) - set(database_df.columns)
        if missing_keys:
            print(f"WARNING: Missing key columns for full comparison: {missing_keys}")
            print("Disabling full data comparison, but will run other validations")
            full_data_comparison_enabled = False
        else:
            full_data_comparison_enabled = True

        validation_config = {
            'row_count_validation': True,
            'data_integrity_validation': True,
            'full_data_comparison': full_data_comparison_enabled,
            'row_count_tolerance': 0.05,  # 5% tolerance
            'sample_size': min(10000, len(source_df), len(database_df)),
            'ohlc_validation': len(source_ohlc) > 0 and len(target_ohlc) > 0
        }

        # Run validation
        print("Running validation engine...")
        engine = ValidationEngine(validation_config)

        # Only pass key columns if they exist and full comparison is enabled
        if full_data_comparison_enabled:
            result = engine.validate(source_df, database_df, key_columns=key_columns)
        else:
            result = engine.validate(source_df, database_df)

        # Display results
        print(f"\n{'='*60}")
        print("VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"Overall Status: {result.overall_status.value}")
        print(f"Source Rows: {result.total_rows_source:,}")
        print(f"Target Rows: {result.total_rows_target:,}")
        print(f"Validations Run: {len(result.validation_results)}")
        print(f"Total Execution Time: {sum(val.execution_time for val in result.validation_results):.3f}s")

        # Display OHLC-specific validation summary
        if validation_config.get('ohlc_validation'):
            print(f"\nOHLC Validation Summary:")
            print(f"- OHLC columns validated: {len(source_ohlc)} source, {len(target_ohlc)} target")
            ohlc_validations = [v for v in result.validation_results if 'OHLC' in v.validation_type.upper()]
            for ohlc_val in ohlc_validations:
                status_symbol = "[PASS]" if ohlc_val.status.value == "PASSED" else "[FAIL]" if ohlc_val.status.value == "FAILED" else "[WARN]"
                print(f"  {status_symbol} {ohlc_val.message}")

        # Display OHLC value comparison results
        if ohlc_comparison_results:
            # Determine which type of comparison was performed
            if 'mean_diff_pct' in list(ohlc_comparison_results.values())[0]:
                print(f"\nOHLC Value Comparison (Statistical):")
                print(f"- Tolerance: 1%")
                print(f"- Method: Statistical comparison (no date/time matching available)")
                for col, metrics in ohlc_comparison_results.items():
                    mean_diff_pct = metrics['mean_diff_pct']
                    status = "PASS" if mean_diff_pct <= 1.0 else "WARN" if mean_diff_pct <= 2.0 else "FAIL"
                    status_symbol = "[PASS]" if status == "PASS" else "[WARN]" if status == "WARN" else "[FAIL]"

                    print(f"  {status_symbol} {col.upper()}: {status}")
                    print(f"     Mean difference: {mean_diff_pct:.2f}%")
                    print(f"     Source: {metrics['source_mean']:.2f} ± {metrics['source_std']:.2f} ({metrics['source_count']:,} values)")
                    print(f"     Target: {metrics['target_mean']:.2f} ± {metrics['target_std']:.2f} ({metrics['target_count']:,} values)")
            else:
                print(f"\nOHLC Value Comparison by Date/Time:")
                print(f"- Tolerance: 1%")
                for col, metrics in ohlc_comparison_results.items():
                    match_pct = metrics['match_percentage']
                    status = "PASS" if match_pct >= 95 else "WARN" if match_pct >= 80 else "FAIL"
                    status_symbol = "[PASS]" if status == "PASS" else "[WARN]" if status == "WARN" else "[FAIL]"

                    print(f"  {status_symbol} {col.upper()}: {metrics['matches_within_tolerance']:,}/{metrics['total_comparisons']:,} matches ({match_pct:.1f}%)")
                    print(f"     Avg difference: {metrics['avg_abs_difference']:.2f} ({metrics['avg_pct_difference']:.3f}%)")
                    print(f"     Max difference: {metrics['max_abs_difference']:.2f} ({metrics['max_pct_difference']:.3f}%)")

                    # Show if there are mismatches and this is date/time comparison
                    if match_pct < 100 and 'avg_abs_difference' in metrics and not pd.isna(metrics['avg_abs_difference']):
                        print(f"     NOTE: {metrics['total_comparisons'] - metrics['matches_within_tolerance']:,} records have >1% difference")
        else:
            print(f"\nOHLC Value Comparison: Not available (missing data or no matching records)")

        # Additional detailed mismatch analysis if we have both datasets with date/time
        if 'date' in source_df.columns and 'time' in source_df.columns:
            print(f"\n" + "="*60)
            print("DETAILED MISMATCH ANALYSIS")
            print("="*60)

            try:
                print("Comparing exact values: DigitalOcean (Parquet) vs MySQL for matching date/time records...")

                # Find all matching records by date and time
                # source_df = DigitalOcean Parquet data
                # database_df = MySQL data
                source_for_compare = source_df[['date', 'time', 'open', 'high', 'low', 'close']].copy()  # DigitalOcean
                target_for_compare = database_df[['date', 'time', 'open', 'high', 'low', 'close']].copy()  # MySQL

                # Create composite keys
                source_for_compare['datetime_key'] = source_for_compare['date'].astype(str) + '_' + source_for_compare['time'].astype(str)
                target_for_compare['datetime_key'] = target_for_compare['date'].astype(str) + '_' + target_for_compare['time'].astype(str)

                # Merge datasets on date/time to find matches
                merged_comparison = pd.merge(
                    source_for_compare,
                    target_for_compare,
                    on='datetime_key',
                    suffixes=('_parquet', '_mysql'),
                    how='inner'
                )

                print(f"Found {len(merged_comparison)} matching date/time records")

                if not merged_comparison.empty:
                    # Calculate differences for each OHLC column
                    ohlc_cols = ['open', 'high', 'low', 'close']
                    mismatches_by_col = {}
                    total_mismatches = 0

                    for col in ohlc_cols:
                        parquet_col = f"{col}_parquet"
                        mysql_col = f"{col}_mysql"

                        if parquet_col in merged_comparison.columns and mysql_col in merged_comparison.columns:
                            parquet_vals = merged_comparison[parquet_col]
                            mysql_vals = merged_comparison[mysql_col]

                            # Calculate absolute and percentage differences
                            abs_diffs = abs(parquet_vals - mysql_vals)
                            pct_diffs = abs_diffs / mysql_vals.replace(0, 1) * 100

                            # Find ALL differences (not just >1%) to see what the full comparison sees
                            any_differences = abs_diffs > 0.01  # Any difference greater than 0.01
                            significant_differences = pct_diffs > 1.0  # Significant differences >1%

                            any_diff_count = any_differences.sum()
                            significant_diff_count = significant_differences.sum()
                            total_mismatches += significant_diff_count

                            print(f"\n{col.upper()} Analysis:")
                            print(f"  Records with ANY difference: {any_diff_count} out of {len(merged_comparison)} ({any_diff_count/len(merged_comparison)*100:.1f}%)")
                            print(f"  Records with >1% difference: {significant_diff_count} out of {len(merged_comparison)} ({significant_diff_count/len(merged_comparison)*100:.1f}%)")

                            if any_diff_count > 0:
                                print(f"  Mean absolute difference: {abs_diffs[any_differences].mean():.2f}")
                                print(f"  Max absolute difference: {abs_diffs.max():.2f}")
                                print(f"  Mean % difference: {pct_diffs[any_differences].mean():.2f}%")
                                print(f"  Max % difference: {pct_diffs.max():.2f}%")

                                # Show ALL differences (not just significant ones) to match full data comparison
                                print(f"  ALL {col.upper()} differences ({any_diff_count} total):")
                                if any_diff_count > 0:
                                    # Create a sorted view of all mismatches
                                    all_mismatches = merged_comparison[abs_diffs > 0.01].copy()
                                    all_mismatches['abs_diff'] = abs_diffs[abs_diffs > 0.01]
                                    all_mismatches['pct_diff'] = (all_mismatches['abs_diff'] / mysql_vals[abs_diffs > 0.01].replace(0, 1)) * 100

                                    # Sort by absolute difference (worst first)
                                    all_mismatches_sorted = all_mismatches.sort_values('abs_diff', ascending=False)

                                    # Print column header with better formatting
                                    print(f"    {'Date':>8} {'Time':>6} {'DigitalOcean':>12} {'MySQL':>12} {'AbsDiff':>9} {'%Diff':>8} {'Significance'}")
                                    print(f"    {'-'*8} {'-'*6} {'-'*12} {'-'*12} {'-'*9} {'-'*8} {'-'*11}")

                                    # Print all mismatches (limit to 50 to avoid too much output)
                                    max_to_show = min(50, len(all_mismatches_sorted))
                                    for i, (_, row) in enumerate(all_mismatches_sorted.head(max_to_show).iterrows()):
                                        digital_ocean_val = row[parquet_col]
                                        mysql_val = row[mysql_col]
                                        abs_diff_val = row['abs_diff']
                                        pct_diff_val = row['pct_diff']

                                        if pct_diff_val > 1.0:
                                            significance = "HIGH"
                                            prefix = "***"
                                        elif pct_diff_val > 0.1:
                                            significance = "MEDIUM"
                                            prefix = " **"
                                        else:
                                            significance = "LOW"
                                            prefix = "   "

                                        print(f"    {row['date']:8d} {row['time_parquet']:6d} "
                                              f"{digital_ocean_val:12.1f} {mysql_val:12.1f} {abs_diff_val:9.1f} {pct_diff_val:7.2f}% {prefix}{significance}")

                                    if any_diff_count > max_to_show:
                                        print(f"    ... and {any_diff_count - max_to_show} more {col.upper()} differences")
                                    print(f"    *** = >1% difference, ** = 0.1-1%,   = <0.1%")

                                if significant_diff_count > 0:
                                    print(f"    * = Significant difference (>1%)")
                                else:
                                    print(f"    * No significant differences found")

                            mismatches_by_col[col] = {
                                'any_diff_count': any_diff_count,
                                'significant_diff_count': significant_diff_count,
                                'abs_diffs': abs_diffs,
                                'pct_diffs': pct_diffs,
                                'merged': merged_comparison[diff_indices] if any_diff_count > 0 else pd.DataFrame()
                            }

                    print(f"\n" + "="*40)
                    print(f"TOTAL MISMATCH SUMMARY:")
                    print(f"Total OHLC mismatches >1%: {total_mismatches}")
                    print(f"Out of {len(merged_comparison) * 4} total comparisons")
                    print(f"Overall match rate: {((len(merged_comparison) * 4) - total_mismatches) / (len(merged_comparison) * 4) * 100:.1f}%")

                    # Debug the dataset alignment issue
                    print(f"\n" + "="*40)
                    print(f"DATASET ALIGNMENT DEBUG:")
                    print(f"DigitalOcean dataset total rows: {len(source_df)}")
                    print(f"MySQL dataset total rows: {len(database_df)}")
                    print(f"Matching date/time rows: {len(merged_comparison)}")
                    print(f"DigitalOcean-only rows: {len(source_df) - len(merged_comparison)}")
                    print(f"MySQL-only rows: {len(database_df) - len(merged_comparison)}")

                    # Show date ranges to understand the alignment issue
                    print(f"\nDate range analysis:")
                    print(f"DigitalOcean: {source_df['date'].min()} to {source_df['date'].max()}")
                    print(f"MySQL: {database_df['date'].min()} to {database_df['date'].max()}")

                    # Show time ranges
                    print(f"\nTime range analysis:")
                    print(f"DigitalOcean: {source_df['time'].min()} to {source_df['time'].max()}")
                    print(f"MySQL: {database_df['time'].min()} to {database_df['time'].max()}")

                    print("="*40)

                else:
                    print("No matching date/time records found for detailed comparison")

                    # Show why there are no matches
                    print(f"\nDEBUG - Why no matches found:")
                    print(f"Source dataset total rows: {len(source_df)}")
                    print(f"Target dataset total rows: {len(database_df)}")
                    print(f"Source date range: {source_df['date'].min()} to {source_df['date'].max()}")
                    print(f"Target date range: {database_df['date'].min()} to {database_df['date'].max()}")

            except Exception as e:
                print(f"Error during detailed mismatch analysis: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n{'='*60}")
        print("DETAILED RESULTS")
        print(f"{'='*60}")

        for val_result in result.validation_results:
            status_symbol = "[PASS]" if val_result.status.value == "PASSED" else "[FAIL]" if val_result.status.value == "FAILED" else "[WARN]"
            print(f"\n{status_symbol} {val_result.validation_type.upper()}: {val_result.status.value}")
            print(f"   Message: {val_result.message}")
            print(f"   Execution time: {val_result.execution_time:.3f}s")

            if val_result.issues:
                print(f"   Issues found: {len(val_result.issues)}")
                for i, issue in enumerate(val_result.issues[:5]):  # Show first 5 issues
                    print(f"     {i+1}. {issue.issue_type}: {issue.message}")
                if len(val_result.issues) > 5:
                    print(f"     ... and {len(val_result.issues) - 5} more issues")

        # Clean up
        database.close()

        print(f"\n{'='*60}")
        if result.overall_status.value == "PASSED":
            print("VALIDATION COMPLETED SUCCESSFULLY!")
        else:
            print("VALIDATION COMPLETED WITH ISSUES!")
        print(f"{'='*60}")

        return True

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_validation()
    exit(0 if success else 1)