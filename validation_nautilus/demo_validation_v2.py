#!/usr/bin/env python3
"""
Demo validation script that shows how to use the interactive validation
Enhanced version with data discovery and selection
"""

import pandas as pd
import re
from datetime import datetime
from collections import Counter
from nautilus_validation.config import ConfigManager
from nautilus_validation.connectors import SpacesConnector, DatabaseConnector
from nautilus_validation.validators import ValidationEngine

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

def extract_symbol_from_filepath(filepath):
    """
    Extract symbol from file path when content extraction fails
    Handles paths like: nautilus_catalog/data/bar/CRUDEOIL-I.NSE-1-TICK-LAST-EXTERNAL/date.parquet
    """
    path_parts = filepath.split('/')

    # Look for symbol-like parts in the path
    for part in path_parts:
        part_upper = part.upper()
        # Check for various crude oil patterns
        if 'CRUDEOIL' in part_upper:
            # Extract the full instrument name
            return part_upper

    # Fallback patterns
    for part in path_parts:
        part_upper = part.upper()
        if any(keyword in part_upper for keyword in ['CRUDE', 'OIL', 'NSE']):
            # Clean up the part to get a meaningful symbol
            if 'NSE' in part_upper:
                return part_upper
            elif 'CRUDE' in part_upper:
                return part_upper

    return None

def parse_parquet_filename(filename, df=None):
    """
    Parse parquet filename to extract symbol, date, and expiry information
    Enhanced to handle timestamp-based filenames like: 2025-06-16T03-30-00-000000000Z_2025-06-16T17-59-58-000000000Z.parquet
    """
    # Remove path and extension
    basename = filename.split('/')[-1].replace('.parquet', '')

    # Pattern 1: Original symbol-based patterns
    patterns = [
        # symbol_date_expiry
        r'^([A-Z]+)_(\d{8})_(\d{8})$',
        # symbol_date
        r'^([A-Z]+)_(\d{8})$',
        # symbol-date-expiry
        r'^([A-Z]+)-(\d{8})-(\d{8})$',
        # symbol-date
        r'^([A-Z]+)-(\d{8})$',
    ]

    for pattern in patterns:
        match = re.match(pattern, basename)
        if match:
            groups = match.groups()
            symbol = groups[0]
            date_str = groups[1]
            expiry = groups[2] if len(groups) > 2 else None

            # Convert date string to proper format
            try:
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                formatted_date = date_obj.strftime('%Y-%m-%d')
            except:
                formatted_date = date_str

            return {
                'symbol': symbol,
                'date': formatted_date,
                'expiry': expiry,
                'original_filename': filename
            }

    # Pattern 2: Timestamp-based filenames like: 2025-06-16T03-30-00-000000000Z_2025-06-16T17-59-58-000000000Z.parquet
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

    # Pattern 3: Fallback - try to extract symbol from path or filename
    symbol = extract_symbol_from_filepath(filename)

    if not symbol:
        # Last resort - first part of filename
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

    # Extract base symbol from complex names
    base_symbol = clean_symbol

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

    # Remove any remaining special characters except common symbol characters
    import re
    base_symbol = re.sub(r'[^A-Z0-9]', '', base_symbol)

    # Determine table type based on context and symbol
    table_type = 'future'  # Default to future as it's most common

    # Check file path for clues about data type
    file_path_lower = file_path.lower()
    if any(indicator in file_path_lower for indicator in ['1-minute', 'min', 'minute']):
        # Could be cash or future, default to future for trading data
        table_type = 'future'
    elif any(indicator in file_path_lower for indicator in ['option', 'call', 'put']):
        # Parse option type from symbol
        if 'CALL' in clean_symbol or 'CE' in clean_symbol:
            table_type = 'call'
        elif 'PUT' in clean_symbol or 'PE' in clean_symbol:
            table_type = 'put'
        else:
            table_type = 'future'
    elif 'tick' in file_path_lower:
        table_type = 'future'

    # Common stock/index pattern detection
    if base_symbol and len(base_symbol) > 2:
        # Check if it's likely a stock (ends with common stock patterns)
        if any(base_symbol.endswith(suffix) for suffix in ['LTD', 'LIMITED', 'CORP', 'IND', 'INFRA']):
            table_type = 'cash'  # Use cash for stocks
        elif base_symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'SENSEX']:
            table_type = 'future'  # Use future for indices

    # Final table name construction
    table_name = f"{base_symbol.lower()}_{table_type}"

    return table_name

def list_available_data(bucket, prefix):
    """List all available data groups in the prefix"""
    print(f"\n{'='*60}")
    print("DISCOVERING AVAILABLE DATA")
    print(f"{'='*60}")
    print(f"Scanning: {bucket}/{prefix}")

    try:
        # Load config and initialize connector
        config = ConfigManager('config.yaml')
        do_config = config.get_digital_ocean_config()
        do_config['bucket_name'] = bucket
        spaces = SpacesConnector(do_config)

        # Get all files in the prefix
        all_objects = spaces.client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=2000
        )

        files = [obj['Key'] for obj in all_objects.get('Contents', [])]
        print(f"Found {len(files)} files in the prefix")

        if not files:
            print("No files found in the specified prefix")
            return None

        # Group files by symbol/instrument
        data_groups = {}

        for file in files:
            if '.parquet' in file.lower():
                # Extract symbol from file path
                path_parts = file.split('/')
                symbol = None

                # Look for symbol-like parts in the path
                for part in path_parts:
                    if any(keyword in part.upper() for keyword in ['NSE', 'BSE', 'FUTURE', 'INDEX', 'MINUTE', 'TICK']):
                        # Extract the base symbol (part before NSE or other patterns)
                        symbol_part = part.split('-NSE')[0] if '-NSE' in part.upper() else part
                        symbol_part = symbol_part.split('.')[0] if '.' in symbol_part else symbol_part
                        if symbol_part and len(symbol_part) > 2:
                            symbol = symbol_part
                            break

                # If no symbol found, use the directory name
                if not symbol:
                    for i, part in enumerate(path_parts[:-1]):  # Exclude filename
                        if part not in ['nautilas-data', 'data', 'bar']:
                            symbol = part
                            break

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
        print(f"\nFound {len(data_groups)} data groups:")
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

def get_user_selection(data_groups):
    """Get user selection of which data to validate"""
    print("\n" + "="*60)
    print("SELECT DATA TO VALIDATE")
    print("="*60)

    while True:
        try:
            selection = input("\nEnter the number of the data group to validate (or 'q' to quit): ").strip()

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
    print("First, we'll discover available data, then select which to validate")

    # Get bucket name
    bucket = input("\nEnter bucket name (default: historical-db-1min): ").strip()
    if not bucket:
        bucket = "historical-db-1min"

    # Get prefix
    print("\nAvailable prefixes you can try:")
    print("- nautilas-data")
    print("- raw/parquet_data/futures/banknifty/2024/01/")
    print("- raw/parquet_data/futures/nifty/2024/01/")
    print("- raw/parquet_data/indices/")
    print("- raw/parquet_data/stocks/")

    prefix = input("\nEnter Digital Ocean prefix/path for parquet files: ").strip()
    if not prefix:
        print("ERROR: Prefix is required")
        return None, None, None, None, None

    # List available data
    data_groups = list_available_data(bucket, prefix)

    if not data_groups:
        print("No data found to validate")
        return None, None, None, None, None

    # Get user selection
    selection_result = get_user_selection(data_groups)

    if not selection_result:
        print("Validation cancelled by user")
        return None, None, None, None, None

    selected_symbol, selected_files, num_files = selection_result

    return bucket, prefix, selected_symbol, selected_files, num_files

def demo_validation():
    """Demo validation with user inputs"""
    # Get user inputs
    bucket, prefix, selected_symbol, selected_files, num_files = get_user_inputs()
    if not bucket or not prefix or not selected_symbol:
        return False

    print(f"\n{'='*60}")
    print("VALIDATION PARAMETERS")
    print(f"{'='*60}")
    print(f"Bucket: {bucket}")
    print(f"Prefix: {prefix}")
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
                    table_columns = [row[0] for _, row in structure_df.iterrows()]
                    print(f"Table columns: {', '.join(table_columns)}")
                except:
                    # Fallback if DESCRIBE doesn't work
                    sample_query = f"SELECT * FROM {table} LIMIT 1"
                    sample_df = pd.read_sql_query(sample_query, conn)
                    table_columns = sample_df.columns.tolist()
                    print(f"Table columns (from sample): {', '.join(table_columns)}")

                # Get date range from Parquet data or parsed filenames
                if 'date' in source_df.columns:
                    min_date = source_df['date'].min()
                    max_date = source_df['date'].max()
                    date_filter = f" WHERE date >= '{min_date}' AND date <= '{max_date}'"
                elif dates_found:
                    min_date = min(dates_found)
                    max_date = max(dates_found)
                    date_filter = f" WHERE date >= '{min_date}' AND date <= '{max_date}'"
                else:
                    date_filter = ""

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
                print(f"Full query: {query}")

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

        # Configure validation with OHLC focus
        print(f"\nConfiguring validation...")

        # Identify OHLC columns in both datasets
        source_ohlc = [col for col in source_df.columns if col.upper() in ['OPEN', 'HIGH', 'LOW', 'CLOSE']]
        target_ohlc = [col for col in database_df.columns if col.upper() in ['OPEN', 'HIGH', 'LOW', 'CLOSE']]

        print(f"Source OHLC columns: {source_ohlc}")
        print(f"Target OHLC columns: {target_ohlc}")

        validation_config = {
            'row_count_validation': True,
            'data_integrity_validation': True,
            'full_data_comparison': True,
            'row_count_tolerance': 0.05,  # 5% tolerance
            'sample_size': min(10000, len(source_df), len(database_df)),
            'ohlc_validation': len(source_ohlc) > 0 and len(target_ohlc) > 0
        }

        # Run validation
        print("Running validation engine...")
        engine = ValidationEngine(validation_config)
        result = engine.validate(source_df, database_df, key_columns=['date', 'time'])

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