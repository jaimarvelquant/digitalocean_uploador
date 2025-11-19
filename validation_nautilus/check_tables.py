#!/usr/bin/env python3
"""
Script to check available tables in the database
"""

import pandas as pd
from sqlalchemy import create_engine, text
from nautilus_validation.config import ConfigManager

def check_database_tables():
    """Check what tables exist in the database"""
    print("Checking available tables in historicaldb database...")

    try:
        # Load configuration
        config = ConfigManager('config.yaml')
        db_config = config.get_database_config()

        # Create database connection
        connection_string = (
            f"mysql+pymysql://{db_config['username']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database_name']}"
            f"?charset={db_config['charset']}"
        )

        engine = create_engine(connection_string)

        # Query to get all tables
        with engine.connect() as conn:
            result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result.fetchall()]

        if tables:
            print(f"\nFound {len(tables)} tables in '{db_config['database_name']}' database:")
            for i, table in enumerate(tables, 1):
                print(f"{i:2d}. {table}")

            # Get row counts for each table
            print(f"\n{'Table Name':<30} {'Row Count':<10}")
            print("-" * 40)

            with engine.connect() as conn:
                for table in tables:
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`"))
                        count = result.scalar()
                        print(f"{table:<30} {count:<10,}")
                    except Exception as e:
                        print(f"{table:<30} {'Error':<10}")
        else:
            print("No tables found in the database.")

        engine.dispose()

    except Exception as e:
        print(f"Error connecting to database: {e}")

if __name__ == "__main__":
    check_database_tables()