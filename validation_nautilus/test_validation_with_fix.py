#!/usr/bin/env python3
"""
Test validation script with the MySQL date reading fix
This version is non-interactive for testing purposes
"""

import pandas as pd
from datetime import datetime
from collections import Counter
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

def test_mysql_data_reading():
    """Test MySQL data reading as it would work in the validation script"""
    print("=" * 60)
    print("TESTING VALIDATION SCRIPT WITH MYSQL FIX")
    print("=" * 60)

    try:
        # Load configuration
        config = ConfigManager('config.yaml')
        database = DatabaseConnector(config.get_database_config())

        # Simulate the validation scenario
        table = 'nifty_future'
        symbols_found = {'NIFTY'}

        # Use actual dates that exist in the database (from our previous test)
        dates_found = {'2025-11-17', '2025-11-18'}  # Convert to database format

        print(f"Testing with table: {table}")
        print(f"Symbols found: {symbols_found}")
        print(f"Original dates found: {dates_found}")

        with database.engine.connect() as conn:
            # Check table structure
            print("\n1. Checking table structure...")
            structure_query = f"DESCRIBE {table}"
            structure_df = pd.read_sql_query(structure_query, conn)
            table_columns = [row[0] for _, row in structure_df.iterrows()]
            print(f"   Table columns: {', '.join(table_columns)}")

            # Build date filter using the conversion function (like in the fixed script)
            print("\n2. Building date filter with conversion...")
            date_filter = ""
            if dates_found:
                # Convert dates to YYMMDD format
                yymmdd_dates = [convert_date_to_yymmdd_format(d) for d in dates_found if d is not None]
                yymmdd_dates = [d for d in yymmdd_dates if d is not None]  # Remove None values

                if yymmdd_dates:
                    min_date = min(yymmdd_dates)
                    max_date = max(yymmdd_dates)
                    date_filter = f" WHERE date >= {min_date} AND date <= {max_date}"
                    print(f"   Converted dates: {yymmdd_dates}")
                    print(f"   Date filter: {date_filter}")
                else:
                    print("   ERROR: Could not convert any dates to YYMMDD format")

            # Add symbol filter
            print("\n3. Adding symbol filter...")
            has_symbol_column = 'symbol' in [col.lower() for col in table_columns]
            if symbols_found and has_symbol_column:
                symbol_list = "', '".join(symbols_found)
                if date_filter:
                    date_filter += f" AND symbol IN ('{symbol_list}')"
                else:
                    date_filter = f" WHERE symbol IN ('{symbol_list}')"
                print(f"   Final filter: {date_filter}")

            # Execute the query (like in the validation script)
            print("\n4. Executing the query...")
            sample_size = 1000
            query = f"SELECT * FROM {table}{date_filter} ORDER BY date, time LIMIT {sample_size}"
            print(f"   Query: {query}")

            database_df = pd.read_sql_query(query, conn)
            print(f"   Database rows read: {len(database_df):,}")

            # Check if we got data
            if database_df.empty:
                print("   WARNING: No data returned. This might be due to:")
                print("   - No data exists for the specified dates")
                print("   - Symbol doesn't match data in the table")
                print("   - Date conversion issue")

                # Let's try a fallback without date filter to verify the table has data
                print("\n   Trying fallback query without date filter...")
                fallback_query = f"SELECT * FROM {table} WHERE symbol IN ('NIFTY') ORDER BY date DESC LIMIT 5"
                fallback_result = pd.read_sql_query(fallback_query, conn)
                print(f"   Fallback query returned {len(fallback_result)} rows")
                if not fallback_result.empty:
                    print("   Recent data sample:")
                    print(fallback_result[['date', 'time', 'symbol', 'open', 'high', 'low', 'close']])
            else:
                print("   SUCCESS: Data retrieved successfully!")

                # Check OHLC columns in database (like in the original script)
                print("\n5. Checking OHLC columns...")
                db_ohlc_cols = [col for col in database_df.columns if col.upper() in ['OPEN', 'HIGH', 'LOW', 'CLOSE']]
                if db_ohlc_cols:
                    print(f"   Database OHLC columns: {', '.join(db_ohlc_cols)}")

                    # Show sample OHLC data
                    print("   Sample OHLC values:")
                    for col in db_ohlc_cols:
                        if col in database_df.columns and not database_df[col].isna().all():
                            print(f"     {col}: min={database_df[col].min()}, max={database_df[col].max()}")

                # Show data types
                print("\n6. Data types:")
                for col in database_df.columns:
                    print(f"   {col}: {database_df[col].dtype}")

                # Show sample rows
                print("\n7. Sample data:")
                print(database_df[['date', 'time', 'symbol', 'open', 'high', 'low', 'close']].head(3))

        database.close()
        print("\n[SUCCESS] MySQL data reading test completed successfully")
        return True

    except Exception as e:
        print(f"\n[ERROR] MySQL data reading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mysql_data_reading()
    exit(0 if success else 1)