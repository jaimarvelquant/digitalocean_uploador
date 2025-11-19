#!/usr/bin/env python3
"""
Test the date conversion fix for MySQL queries
"""

import pandas as pd
from datetime import datetime
from nautilus_validation.config import ConfigManager
from nautilus_validation.connectors import DatabaseConnector

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

def test_date_conversion():
    """Test date conversion and MySQL query with proper date format"""
    print("=" * 60)
    print("TESTING DATE CONVERSION FIX")
    print("=" * 60)

    # Test the conversion function
    print("1. Testing date conversion function:")
    test_dates = ['2024-05-20', '2024-05-21', '2024-12-31', '2023-01-01']
    for date_str in test_dates:
        converted = convert_date_to_yymmdd_format(date_str)
        print(f"   {date_str} -> {converted}")

    # Test with actual database
    print("\n2. Testing with actual database:")
    try:
        config = ConfigManager('config.yaml')
        database = DatabaseConnector(config.get_database_config())

        with database.engine.connect() as conn:
            # Get date range from database
            date_range_query = f"SELECT MIN(date) as min_date, MAX(date) as max_date FROM nifty_future"
            date_range_result = pd.read_sql_query(date_range_query, conn)
            min_date = date_range_result.iloc[0]['min_date']
            max_date = date_range_result.iloc[0]['max_date']
            print(f"   Database date range: {min_date} to {max_date}")

            # Test query with converted dates
            test_dates = {'2024-11-15', '2024-11-16'}  # Recent dates
            converted_dates = []
            for date_val in test_dates:
                converted = convert_date_to_yymmdd_format(date_val)
                if converted:
                    converted_dates.append(converted)

            if converted_dates:
                query_min = min(converted_dates)
                query_max = max(converted_dates)

                print(f"   Testing query with converted dates {query_min} to {query_max}")

                # Test query
                test_query = f"SELECT * FROM nifty_future WHERE date >= {query_min} AND date <= {query_max} ORDER BY date, time LIMIT 10"
                print(f"   Query: {test_query}")

                result_df = pd.read_sql_query(test_query, conn)
                print(f"   Rows returned: {len(result_df)}")

                if not result_df.empty:
                    print("   Sample data:")
                    print(result_df[['date', 'time', 'symbol', 'open', 'high', 'low', 'close']].head())
                else:
                    # Try with a wider date range to see if there's any data
                    print("   No data found, trying with recent dates from database...")
                    recent_dates_query = f"SELECT DISTINCT date FROM nifty_future ORDER BY date DESC LIMIT 3"
                    recent_dates = pd.read_sql_query(recent_dates_query, conn)
                    print(f"   Recent database dates: {recent_dates['date'].tolist()}")

                    if len(recent_dates) >= 2:
                        max_date = recent_dates['date'].iloc[0]
                        min_date = recent_dates['date'].iloc[1]
                        wide_query = f"SELECT * FROM nifty_future WHERE date >= {min_date} AND date <= {max_date} ORDER BY date, time LIMIT 10"
                        wide_result = pd.read_sql_query(wide_query, conn)
                        print(f"   Rows returned with wide range: {len(wide_result)}")
                        if not wide_result.empty:
                            print("   Sample data from wide query:")
                            print(wide_result[['date', 'time', 'symbol', 'open', 'high', 'low', 'close']].head())

        database.close()
        print("\n[SUCCESS] Date conversion test completed")

    except Exception as e:
        print(f"\n[ERROR] Date conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = test_date_conversion()
    exit(0 if success else 1)