#!/usr/bin/env python3
"""
Test script to check MySQL data reading specifically in the validation context
"""

import pandas as pd
from datetime import datetime
from nautilus_validation.config import ConfigManager
from nautilus_validation.connectors import DatabaseConnector

def test_mysql_reading():
    """Test MySQL data reading similar to how it's done in the validation script"""
    print("=" * 60)
    print("TESTING MYSQL DATA READING FOR VALIDATION")
    print("=" * 60)

    try:
        # Load configuration
        config = ConfigManager('config.yaml')
        database = DatabaseConnector(config.get_database_config())

        # Test table reading like in the validation script
        table = 'nifty_future'  # Use a known table from our test
        dates_found = {'2024-05-20', '2024-05-21'}  # Sample dates
        symbols_found = {'NIFTY'}  # Sample symbols

        print(f"Testing table: {table}")

        with database.engine.connect() as conn:
            # Test table structure checking (like line 803-814 in original)
            print("\n1. Testing table structure...")
            try:
                structure_query = f"DESCRIBE {table}"
                structure_df = pd.read_sql_query(structure_query, conn)
                table_columns = [row[0] for _, row in structure_df.iterrows()]
                print(f"   Table columns: {', '.join(table_columns)}")
            except Exception as e:
                print(f"   Error with DESCRIBE: {e}")
                # Fallback
                sample_query = f"SELECT * FROM {table} LIMIT 1"
                sample_df = pd.read_sql_query(sample_query, conn)
                table_columns = sample_df.columns.tolist()
                print(f"   Table columns (from sample): {', '.join(table_columns)}")

            # Test date filter building (like line 817-826)
            print("\n2. Testing date filter building...")
            if dates_found:
                min_date = min(dates_found)
                max_date = max(dates_found)
                date_filter = f" WHERE date >= '{min_date}' AND date <= '{max_date}'"
                print(f"   Date filter: {date_filter}")
            else:
                date_filter = ""
                print("   No date filter applied")

            # Test symbol filter building (like line 828-835)
            print("\n3. Testing symbol filter building...")
            has_symbol_column = 'symbol' in [col.lower() for col in table_columns]
            if symbols_found and has_symbol_column:
                symbol_list = "', '".join(symbols_found)
                if date_filter:
                    date_filter += f" AND symbol IN ('{symbol_list}')"
                else:
                    date_filter = f" WHERE symbol IN ('{symbol_list}')"
                print(f"   Added symbol filter: {date_filter}")
            else:
                print(f"   Symbol column exists: {has_symbol_column}")
                if not has_symbol_column:
                    print("   No symbol column found in table")

            # Test query construction and execution (like line 837-842)
            print("\n4. Testing query execution...")

            # First try a simple query
            simple_query = f"SELECT COUNT(*) as row_count FROM {table}"
            count_result = pd.read_sql_query(simple_query, conn)
            total_rows = count_result.iloc[0]['row_count']
            print(f"   Total rows in table: {total_rows:,}")

            # Check actual date format
            date_query = f"SELECT DISTINCT date FROM {table} ORDER BY date DESC LIMIT 10"
            date_result = pd.read_sql_query(date_query, conn)
            print(f"   Recent dates in database: {date_result['date'].tolist()}")

            # Check date range
            date_range_query = f"SELECT MIN(date) as min_date, MAX(date) as max_date FROM {table}"
            date_range_result = pd.read_sql_query(date_range_query, conn)
            min_date = date_range_result.iloc[0]['min_date']
            max_date = date_range_result.iloc[0]['max_date']
            print(f"   Date range: {min_date} to {max_date}")

            # Now test the more complex query like in the original script
            sample_size = 1000
            complex_query = f"SELECT * FROM {table}{date_filter} ORDER BY date, time LIMIT {sample_size}"
            print(f"   Query: {complex_query}")

            database_df = pd.read_sql_query(complex_query, conn)
            print(f"   Database rows read: {len(database_df):,}")

            # Test OHLC columns detection (like line 844-847)
            print("\n5. Testing OHLC columns detection...")
            db_ohlc_cols = [col for col in database_df.columns if col.upper() in ['OPEN', 'HIGH', 'LOW', 'CLOSE']]
            if db_ohlc_cols:
                print(f"   Database OHLC columns: {', '.join(db_ohlc_cols)}")

                # Show sample data for OHLC columns
                print("   Sample OHLC data:")
                for col in db_ohlc_cols:
                    if col in database_df.columns:
                        print(f"     {col}: min={database_df[col].min()}, max={database_df[col].max()}")
            else:
                print("   No OHLC columns found")

            # Show data types to debug potential issues
            print("\n6. Data types:")
            for col in database_df.columns[:10]:  # Show first 10 columns
                dtype = database_df[col].dtype
                print(f"   {col}: {dtype}")

            # Show sample rows
            print("\n7. Sample data:")
            print(database_df.head(3).to_string())

        # Clean up
        database.close()
        print("\n[SUCCESS] MySQL data reading test completed successfully")

    except Exception as e:
        print(f"\n[ERROR] MySQL data reading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = test_mysql_reading()
    exit(0 if success else 1)