#!/usr/bin/env python3
"""
Test script to check MySQL connection and data reading
"""

import pandas as pd
from nautilus_validation.config import ConfigManager
from nautilus_validation.connectors import DatabaseConnector

def test_mysql_connection():
    """Test MySQL database connection and basic queries"""
    print("=" * 60)
    print("TESTING MYSQL CONNECTION")
    print("=" * 60)

    try:
        # Load configuration
        config = ConfigManager('config.yaml')
        db_config = config.get_database_config()

        print("Database config:")
        for key, value in db_config.items():
            if key == 'password':
                print(f"  {key}: {'*' * len(str(value))}")
            else:
                print(f"  {key}: {value}")

        # Initialize database connector
        print("\nInitializing database connection...")
        database = DatabaseConnector(db_config)

        # Test basic connection
        print("[SUCCESS] Database connection successful")

        # Test basic query
        print("\nTesting basic query...")
        with database.engine.connect() as conn:
            result = pd.read_sql_query("SELECT 1 as test", conn)
            print("[SUCCESS] Basic query successful")

        # List available tables
        print("\nListing available tables...")
        try:
            with database.engine.connect() as conn:
                tables_df = pd.read_sql_query("SHOW TABLES", conn)
                print(f"Found {len(tables_df)} tables:")
                for idx, row in tables_df.iterrows():
                    table_name = row.iloc[0]
                    print(f"  {idx + 1}. {table_name}")
        except Exception as e:
            print(f"Error listing tables: {e}")

        # Test table structure check for a likely table
        test_tables = ['nifty_future', 'banknifty_future', 'crudeoil_future']

        for table in test_tables:
            print(f"\nTesting table: {table}")
            try:
                with database.engine.connect() as conn:
                    # Check if table exists
                    structure_query = f"DESCRIBE {table}"
                    structure_df = pd.read_sql_query(structure_query, conn)
                    print(f"[SUCCESS] Table '{table}' exists with {len(structure_df)} columns:")
                    for _, row in structure_df.iterrows():
                        print(f"    - {row[0]} ({row[1]})")

                    # Test reading a few rows
                    sample_query = f"SELECT * FROM {table} LIMIT 5"
                    sample_df = pd.read_sql_query(sample_query, conn)
                    print(f"[SUCCESS] Successfully read {len(sample_df)} sample rows from '{table}'")

                    # Show column names
                    print(f"  Columns: {', '.join(sample_df.columns)}")

                    # Check if table has date and time columns
                    if 'date' in sample_df.columns:
                        print(f"  Date range: {sample_df['date'].min()} to {sample_df['date'].max()}")
                    if 'time' in sample_df.columns:
                        print(f"  Time range: {sample_df['time'].min()} to {sample_df['time'].max()}")

                    break  # Stop at first successful table

            except Exception as e:
                print(f"[ERROR] Error accessing table '{table}': {e}")

        # Clean up
        database.close()
        print("\n[SUCCESS] Database connection closed successfully")

    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = test_mysql_connection()
    exit(0 if success else 1)