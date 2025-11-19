#!/usr/bin/env python3
"""
Interactive validation script that takes bucket and prefix as input
and validates data against MySQL database
"""

import pandas as pd
import sys
from nautilus_validation.config import ConfigManager
from nautilus_validation.connectors import SpacesConnector, DatabaseConnector
from nautilus_validation.validators import ValidationEngine

def get_user_input():
    """Get bucket and prefix from user input"""
    print("=" * 60)
    print("NAUTILUS DATA VALIDATION SYSTEM")
    print("=" * 60)

    # Get bucket name
    bucket = input("\nEnter bucket name (default: trading-data): ").strip()
    if not bucket:
        bucket = "trading-data"

    # Get prefix
    prefix = input("Enter prefix/path (e.g., raw/parquet_data/futures/banknifty/2024/01/): ").strip()
    if not prefix:
        print("ERROR: Prefix is required")
        return None, None

    # Get table name
    table = input("Enter MySQL table name (e.g., banknifty_future): ").strip()
    if not table:
        print("ERROR: Table name is required")
        return None, None

    # Get key columns
    key_columns_input = input("Enter key columns (comma-separated, default: date,time): ").strip()
    if not key_columns_input:
        key_columns = ['date', 'time']
    else:
        key_columns = [col.strip() for col in key_columns_input.split(',')]

    return bucket, prefix, table, key_columns

def confirm_validation_parameters(bucket, prefix, table, key_columns):
    """Confirm validation parameters with user"""
    print("\n" + "=" * 60)
    print("VALIDATION PARAMETERS")
    print("=" * 60)
    print(f"Bucket: {bucket}")
    print(f"Prefix: {prefix}")
    print(f"Table: {table}")
    print(f"Key Columns: {', '.join(key_columns)}")

    confirm = input("\nProceed with validation? (y/N): ").strip().lower()
    return confirm in ['y', 'yes']

def interactive_validation():
    """Run interactive validation with user input"""
    try:
        # Get user input
        bucket, prefix, table, key_columns = get_user_input()
        if not bucket or not prefix or not table:
            return False

        # Confirm parameters
        if not confirm_validation_parameters(bucket, prefix, table, key_columns):
            print("Validation cancelled by user")
            return False

        print("\n" + "=" * 60)
        print("STARTING VALIDATION")
        print("=" * 60)

        # Load configuration
        print("Loading configuration...")
        config = ConfigManager('config.yaml')

        # Initialize connectors
        print("Initializing connectors...")
        spaces = SpacesConnector(config.get_digital_ocean_config())
        database = DatabaseConnector(config.get_database_config())

        # List available Parquet files
        print(f"\nListing files in {bucket}/{prefix}")
        try:
            # Update bucket in config
            do_config = config.get_digital_ocean_config()
            do_config['bucket_name'] = bucket
            spaces = SpacesConnector(do_config)

            parquet_files = spaces.list_parquet_files(prefix)
            if not parquet_files:
                print(f"No Parquet files found in {prefix}")
                return False

            print(f"Found {len(parquet_files)} Parquet files")

            # Ask user for number of files to process
            max_files = min(len(parquet_files), 10)  # Default to max 10 files
            num_files_input = input(f"Number of files to process (1-{len(parquet_files)}, default: {max_files}): ").strip()

            if not num_files_input:
                num_files = max_files
            else:
                num_files = min(int(num_files_input), len(parquet_files))

            print(f"Processing {num_files} files...")

        except Exception as e:
            print(f"Error accessing files: {e}")
            return False

        # Read Parquet data
        parquet_dfs = []
        total_rows = 0

        for i, file in enumerate(parquet_files[:num_files]):
            print(f"Reading file {i+1}/{num_files}: {file}")
            try:
                df = spaces.read_parquet_file(file)
                parquet_dfs.append(df)
                file_rows = len(df)
                total_rows += file_rows
                print(f"   Read {file_rows:,} rows")
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
                # Get date range from Parquet data
                if 'date' in source_df.columns:
                    min_date = source_df['date'].min()
                    max_date = source_df['date'].max()
                    date_filter = f" WHERE date >= '{min_date}' AND date <= '{max_date}'"
                else:
                    date_filter = ""

                # Construct query
                query = f"SELECT * FROM {table}{date_filter} LIMIT {len(source_df) + 1000}"
                print(f"Query: {query[:100]}...")

                database_df = pd.read_sql_query(query, conn)
                print(f"Database rows read: {len(database_df):,}")

        except Exception as e:
            print(f"Error reading database: {e}")
            return False

        # Configure validation
        print(f"\nConfiguring validation...")
        validation_config = {
            'row_count_validation': True,
            'data_integrity_validation': True,
            'full_data_comparison': True,
            'row_count_tolerance': 0.05,  # 5% tolerance
            'sample_size': min(10000, len(source_df), len(database_df))
        }

        # Run validation
        print("Running validation engine...")
        engine = ValidationEngine(validation_config)
        result = engine.validate(source_df, database_df, key_columns=key_columns)

        # Display results
        print(f"\n{'='*60}")
        print("VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"Overall Status: {result.overall_status.value}")
        print(f"Source Rows: {result.total_rows_source:,}")
        print(f"Target Rows: {result.total_rows_target:,}")
        print(f"Validations Run: {len(result.validation_results)}")
        print(f"Total Execution Time: {sum(val.execution_time for val in result.validation_results):.3f}s")

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

    except KeyboardInterrupt:
        print(f"\n\nValidation interrupted by user")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = interactive_validation()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\nGoodbye!")
        exit(1)