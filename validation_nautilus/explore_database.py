#!/usr/bin/env python3
"""
Database exploration script to understand table structure and symbol naming conventions
"""

import pandas as pd
from nautilus_validation.config import ConfigManager
from nautilus_validation.connectors import DatabaseConnector

def explore_database():
    """Explore database structure and symbol naming"""
    print("=" * 60)
    print("DATABASE EXPLORATION TOOL")
    print("=" * 60)

    try:
        # Load configuration
        print("Loading configuration...")
        config = ConfigManager('config.yaml')

        # Initialize database connector
        print("Connecting to database...")
        database = DatabaseConnector(config.get_database_config())

        with database.engine.connect() as conn:
            print(f"\n✅ Connected to database successfully")

            # 1. Get basic database info
            print(f"\n{'='*60}")
            print("DATABASE INFORMATION")
            print(f"{'='*60}")

            try:
                # Get current database name
                db_result = pd.read_sql_query("SELECT DATABASE() as current_db", conn)
                current_db = db_result.iloc[0]['current_db']
                print(f"Current Database: {current_db}")
            except:
                print("Could not determine current database name")

            # 2. List all tables
            print(f"\n{'='*60}")
            print("ALL TABLES IN DATABASE")
            print(f"{'='*60}")

            try:
                tables_query = "SHOW TABLES"
                tables_df = pd.read_sql_query(tables_query, conn)
                print(f"Found {len(tables_df)} tables:")

                table_names = []
                for idx, row in tables_df.iterrows():
                    table_name = row.iloc[0]  # First column contains table name
                    table_names.append(table_name)
                    print(f"  {idx+1:2d}. {table_name}")

            except Exception as e:
                print(f"Error listing tables: {e}")
                return

            # 3. Look for tables with common patterns
            print(f"\n{'='*60}")
            print("SEARCHING FOR RELEVANT TABLES")
            print(f"{'='*60}")

            # Common patterns to search for
            search_patterns = ['nifty', 'bank', 'crude', 'oil', 'stock', 'future', 'index', '3i', 'infotech']

            relevant_tables = []
            for table_name in table_names:
                table_lower = table_name.lower()
                for pattern in search_patterns:
                    if pattern in table_lower:
                        relevant_tables.append(table_name)
                        break

            if relevant_tables:
                print(f"Found {len(relevant_tables)} potentially relevant tables:")
                for i, table in enumerate(relevant_tables, 1):
                    print(f"  {i}. {table}")
            else:
                print("No tables with common trading patterns found")

            # 4. Examine structure of relevant tables
            if relevant_tables:
                print(f"\n{'='*60}")
                print("DETAILED TABLE ANALYSIS")
                print(f"{'='*60}")

                for table_name in relevant_tables[:5]:  # Limit to first 5 to avoid too much output
                    print(f"\n--- Table: {table_name} ---")

                    try:
                        # Get table structure
                        desc_query = f"DESCRIBE `{table_name}`"
                        desc_df = pd.read_sql_query(desc_query, conn)

                        print("Columns:")
                        for idx, row in desc_df.iterrows():
                            col_name = row['Field']
                            col_type = row['Type']
                            nullable = "NULL" if row['Null'] == "YES" else "NOT NULL"
                            key = row['Key'] if row['Key'] else ""
                            print(f"  {col_name:20s} {col_type:20s} {nullable:8s} {key}")

                        # Get row count
                        count_query = f"SELECT COUNT(*) as row_count FROM `{table_name}`"
                        count_result = pd.read_sql_query(count_query, conn)
                        row_count = count_result.iloc[0]['row_count']
                        print(f"Total Rows: {row_count:,}")

                        # Check for symbol-like columns and show sample data
                        symbol_columns = []
                        for idx, row in desc_df.iterrows():
                            col_name = row['Field'].lower()
                            if any(keyword in col_name for keyword in ['symbol', 'instrument', 'ticker', 'code', 'name']):
                                symbol_columns.append(row['Field'])

                        if symbol_columns:
                            print(f"Potential symbol columns: {', '.join(symbol_columns)}")

                            # Show sample data for symbol columns
                            sample_query = f"SELECT {', '.join(symbol_columns)} FROM `{table_name}` LIMIT 5"
                            try:
                                sample_df = pd.read_sql_query(sample_query, conn)
                                print("Sample data:")
                                print(sample_df.to_string(index=False))
                            except Exception as sample_e:
                                print(f"Could not get sample data: {sample_e}")

                        # Check for OHLC columns
                        ohlc_columns = []
                        for idx, row in desc_df.iterrows():
                            col_name = row['Field'].lower()
                            if col_name in ['open', 'high', 'low', 'close']:
                                ohlc_columns.append(row['Field'])

                        if ohlc_columns:
                            print(f"OHLC columns found: {', '.join(ohlc_columns)}")

                        # Check for date/time columns
                        datetime_columns = []
                        for idx, row in desc_df.iterrows():
                            col_name = row['Field'].lower()
                            if any(keyword in col_name for keyword in ['date', 'time', 'timestamp']):
                                datetime_columns.append(row['Field'])

                        if datetime_columns:
                            print(f"Date/Time columns: {', '.join(datetime_columns)}")

                            # Show date range
                            if 'date' in datetime_columns:
                                try:
                                    range_query = f"SELECT MIN(date) as min_date, MAX(date) as max_date FROM `{table_name}`"
                                    range_df = pd.read_sql_query(range_query, conn)
                                    min_date = range_df.iloc[0]['min_date']
                                    max_date = range_df.iloc[0]['max_date']
                                    print(f"Date range: {min_date} to {max_date}")
                                except:
                                    pass

                    except Exception as table_e:
                        print(f"Error analyzing table {table_name}: {table_e}")

            # 5. Look for 3IINFOTECH specifically
            print(f"\n{'='*60}")
            print("SEARCHING FOR 3IINFOTECH RELATED DATA")
            print(f"{'='*60}")

            threei_tables = []
            for table_name in table_names:
                if '3i' in table_name.lower() or 'infotech' in table_name.lower():
                    threei_tables.append(table_name)

            if threei_tables:
                print(f"Found {len(threei_tables)} tables with 3IINFOTECH:")
                for table in threei_tables:
                    print(f"  - {table}")

                    # Show structure of the first 3IINFOTECH table
                    if threei_tables.index(table) == 0:
                        try:
                            print(f"\nDetailed structure for {table}:")
                            desc_query = f"DESCRIBE `{table}`"
                            desc_df = pd.read_sql_query(desc_query, conn)
                            for idx, row in desc_df.iterrows():
                                print(f"  {row['Field']:20s} {row['Type']:20s}")

                            # Show sample data
                            sample_query = f"SELECT * FROM `{table}` LIMIT 3"
                            sample_df = pd.read_sql_query(sample_query, conn)
                            print(f"\nSample data from {table}:")
                            print(sample_df.to_string(index=False))
                        except Exception as e:
                            print(f"Error getting details: {e}")
            else:
                print("No tables found with 3IINFOTECH in the name")

                # Try searching in table contents
                print("\nSearching for 3IINFOTECH in table contents...")
                for table_name in table_names[:10]:  # Check first 10 tables
                    try:
                        # Try to find if there's a symbol column and search for 3IINFOTECH
                        columns_query = f"SHOW COLUMNS FROM `{table_name}`"
                        cols_df = pd.read_sql_query(columns_query, conn)

                        symbol_col = None
                        for idx, row in cols_df.iterrows():
                            if 'symbol' in row['Field'].lower() or 'instrument' in row['Field'].lower():
                                symbol_col = row['Field']
                                break

                        if symbol_col:
                            search_query = f"SELECT COUNT(*) as count FROM `{table_name}` WHERE `{symbol_col}` LIKE '%3I%' OR `{symbol_col}` LIKE '%infotech%'"
                            search_result = pd.read_sql_query(search_query, conn)
                            count = search_result.iloc[0]['count']
                            if count > 0:
                                print(f"Found {count} matching records in table: {table_name}")

                                # Show the actual symbols
                                symbols_query = f"SELECT DISTINCT `{symbol_col}` FROM `{table_name}` WHERE `{symbol_col}` LIKE '%3I%' OR `{symbol_col}` LIKE '%infotech%' LIMIT 5"
                                symbols_df = pd.read_sql_query(symbols_query, conn)
                                print(f"Symbol samples: {symbols_df[symbol_col].tolist()}")
                    except:
                        continue

        # Clean up
        database.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n{'='*60}")
    print("DATABASE EXPLORATION COMPLETED")
    print(f"{'='*60}")

if __name__ == "__main__":
    explore_database()