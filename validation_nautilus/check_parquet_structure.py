#!/usr/bin/env python3
"""
Script to examine Parquet file structure and compare with database table
"""

import pandas as pd
from nautilus_validation.config import ConfigManager
from nautilus_validation.connectors import SpacesConnector, DatabaseConnector

def examine_data_structures():
    """Examine Parquet and database table structures"""
    print("Examining data structures...")

    try:
        # Load configuration
        config = ConfigManager('config.yaml')

        # Initialize connectors
        spaces = SpacesConnector(config.get_digital_ocean_config())
        database = DatabaseConnector(config.get_database_config())

        # Get Parquet file structure
        parquet_file = "raw/parquet_data/futures/banknifty/2024/01/banknifty_future_01012024.parquet"
        print(f"\n1. Examining Parquet file: {parquet_file}")

        try:
            parquet_df = spaces.read_parquet_file(parquet_file)
            print(f"Parquet file shape: {parquet_df.shape}")
            print(f"Parquet columns: {list(parquet_df.columns)}")
            print(f"Sample data:")
            print(parquet_df.head(3))
            print(f"Data types:")
            print(parquet_df.dtypes)
        except Exception as e:
            print(f"Error reading Parquet file: {e}")
            return

        # Get database table structure
        table_name = "banknifty_future"
        print(f"\n2. Examining database table: {table_name}")

        try:
            table_info = database.get_table_info(table_name)
            print(f"Database table row count: {table_info['row_count']:,}")
            print(f"Database columns: {table_info['column_names']}")
            print(f"Column details:")
            for col_name, col_info in table_info['columns'].items():
                print(f"  {col_name}: {col_info['type']} (nullable: {col_info['nullable']})")

            # Get sample data from database
            print(f"\n3. Getting sample database data...")
            with database.engine.connect() as conn:
                sample_df = pd.read_sql_query(f"SELECT * FROM `{table_name}` LIMIT 3", conn)
                print(f"Sample database data:")
                print(sample_df)

        except Exception as e:
            print(f"Error accessing database table: {e}")
            return

        # Compare structures
        print(f"\n4. Structure Comparison:")
        parquet_cols = set(parquet_df.columns)
        db_cols = set(table_info['column_names'])

        common_cols = parquet_cols & db_cols
        parquet_only = parquet_cols - db_cols
        db_only = db_cols - parquet_cols

        print(f"Common columns: {sorted(common_cols)}")
        print(f"Only in Parquet: {sorted(parquet_only)}")
        print(f"Only in Database: {sorted(db_only)}")

        # Clean up
        database.close()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    examine_data_structures()