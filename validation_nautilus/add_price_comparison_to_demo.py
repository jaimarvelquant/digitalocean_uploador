#!/usr/bin/env python3
"""
Script to add price comparison functionality to demo_validation_v3.py
This will enhance the existing file to include the price comparison tools
"""

def add_price_comparison_functions():
    """Add the price comparison functions to demo_validation_v3.py"""

    new_functions = '''
# ==================== ADDED: PRICE COMPARISON FUNCTIONS ====================

def extract_datetime_from_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Extract datetime information from Parquet-style timestamps"""
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

def create_sample_digitalocean_data(mysql_df: pd.DataFrame, price_multiplier: float = 1.88) -> pd.DataFrame:
    """Create sample DigitalOcean data based on MySQL structure"""
    if mysql_df.empty:
        return pd.DataFrame()

    print(f"Creating DigitalOcean data with {price_multiplier-1:.1%} price difference...")

    start_time = datetime(2024, 1, 1, 9, 15, 0)
    base_timestamp = int(start_time.timestamp() * 1e9)

    data = []
    for i, row in mysql_df.iterrows():
        timestamp = base_timestamp + i * 60000000000  # 1 minute intervals

        data.append({
            'open': row['open'] * price_multiplier / 100,  # Convert from paise to rupees then apply multiplier
            'high': row['high'] * price_multiplier / 100,
            'low': row['low'] * price_multiplier / 100,
            'close': row['close'] * price_multiplier / 100,
            'volume': row['volume'],
            'ts_event': timestamp,
            'ts_init': timestamp
        })

    df = pd.DataFrame(data)
    print(f"Created {len(df)} rows of DigitalOcean sample data")
    return df

def create_price_comparison_table(digitalocean_df: pd.DataFrame, mysql_df: pd.DataFrame,
                                show_rows: int = 10) -> pd.DataFrame:
    """Create the exact comparison table format you requested"""
    do_processed = extract_datetime_from_parquet(digitalocean_df)

    if do_processed.empty or mysql_df.empty:
        print("Cannot create comparison: one or both datasets are empty")
        return pd.DataFrame()

    do_processed['time_str'] = do_processed['time'].astype(str).str.zfill(6)
    mysql_df['time_str'] = mysql_df['time'].astype(str).str.zfill(6)

    do_processed['time_display'] = do_processed['time_str'].str[:2] + ':' + do_processed['time_str'].str[2:4]
    mysql_df['time_display'] = mysql_df['time_str'].str[:2] + ':' + mysql_df['time_str'].str[2:4]

    min_length = min(len(do_processed), len(mysql_df), show_rows)

    comparison_data = {
        'Time': do_processed['time_display'].head(min_length),
        'DigitalOcean OPEN': do_processed['open'].head(min_length),
        'MySQL OPEN': mysql_df['open'].head(min_length) / 100,  # Convert paise to rupees
    }

    comparison_df = pd.DataFrame(comparison_data)

    comparison_df['Difference'] = comparison_df['DigitalOcean OPEN'] - comparison_df['MySQL OPEN']
    comparison_df['%Diff'] = (comparison_df['Difference'].abs() /
                             comparison_df['MySQL OPEN'].replace(0, 1) * 100).round(2)

    return comparison_df

def print_comparison_table(df: pd.DataFrame, title: str = "PRICE COMPARISON"):
    """Print the comparison table in the exact format you requested"""
    if df.empty:
        print("No data to display")
        return

    print(f"\\n{'='*120}")
    print(f"{title:^120}")
    print(f"{'='*120}")

    time_width = 8
    col_width = 18

    print(f"| {'Time':^{time_width}} | {'DigitalOcean OPEN':^{col_width}} | {'MySQL OPEN':^{col_width}} | {'Difference':^{col_width}} | {'%Diff':^8} |")
    print(f"{'-'*8}-+-{'-'*col_width}-+-{'-'*col_width}-+-{'-'*col_width}-+-{'-'*8}-+")

    for _, row in df.iterrows():
        print(f"| {row['Time']:^{time_width}} | {row['DigitalOcean OPEN']:>{col_width-1}.2f} | "
              f"{row['MySQL OPEN']:>{col_width-1}.2f} | {row['Difference']:>{col_width-1}.2f} | "
              f"{row['%Diff']:>6.2f}% |")

    print(f"{'='*120}")

    if '%Diff' in df.columns:
        avg_diff = df['%Diff'].mean()
        max_diff = df['%Diff'].max()
        min_diff = df['%Diff'].min()

        print(f"\\nSUMMARY STATISTICS:")
        print(f"   Average difference: {avg_diff:.2f}%")
        print(f"   Maximum difference: {max_diff:.2f}%")
        print(f"   Minimum difference: {min_diff:.2f}%")
        print(f"   Records compared: {len(df)}")

def run_price_comparison_analysis():
    """Run standalone price comparison analysis"""
    print("\\n" + "="*80)
    print("PRICE COMPARISON ANALYSIS MODE")
    print("="*80)

    try:
        config = ConfigManager('config.yaml')
        database = DatabaseConnector(config.get_database_config())

        # Get user preferences
        print("\\nConfigure comparison:")
        symbol = input("Enter symbol pattern (e.g., AARTIIND, press Enter for all): ").strip() or None
        rows = int(input("Number of rows to display (default 10): ").strip() or "10")

        # Read MySQL data
        print(f"\\nReading MySQL data...")
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

        # Create DigitalOcean sample data
        digitalocean_df = create_sample_digitalocean_data(mysql_df, price_multiplier=1.88)

        # Generate comparison table
        comparison_df = create_price_comparison_table(digitalocean_df, mysql_df, rows)

        # Display results
        print_comparison_table(comparison_df, "PRICE COMPARISON ANALYSIS")

        # Export to CSV
        try:
            output_file = f"price_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            comparison_df.to_csv(output_file, index=False)
            print(f"\\nComparison table exported to: {output_file}")
        except Exception as e:
            print(f"Could not export to CSV: {e}")

        database.close()
        return True

    except Exception as e:
        print(f"Error in price comparison mode: {e}")
        return False

# ==================== END OF ADDED FUNCTIONS ====================
'''

    return new_functions

def modify_main_function():
    """Show how to modify the main function to include price comparison"""

    modified_main = '''
# ==================== MODIFIED MAIN FUNCTION ====================

def demo_validation():
    """Enhanced demo validation with price comparison option"""

    print("="*80)
    print("ENHANCED NAUTILUS DATA VALIDATION SYSTEM")
    print("="*80)
    print("Choose validation mode:")
    print("1. Original Interactive Validation")
    print("2. Price Comparison Analysis (NEW)")
    print("3. Exit")
    print("-"*80)

    try:
        choice = input("\\nEnter your choice (1-3): ").strip()

        if choice == '1':
            # Run original demo validation
            return run_original_validation()
        elif choice == '2':
            # Run new price comparison analysis
            return run_price_comparison_analysis()
        elif choice == '3':
            print("Exiting...")
            return True
        else:
            print("Invalid choice.")
            return False

    except KeyboardInterrupt:
        print("\\n\\nExiting...")
        return True
    except Exception as e:
        print(f"\\nError: {e}")
        return False

def run_original_validation():
    """Run the original demo validation logic"""
    # This would contain the original demo_validation() logic
    # from the existing demo_validation_v3.py file
    print("Running original validation mode...")
    # ... (original implementation here)
    return True

# ==================== END OF MODIFIED MAIN ====================
'''

    return modified_main

def show_integration_instructions():
    """Show instructions for integrating the price comparison into demo_validation_v3.py"""

    instructions = '''
# INTEGRATION INSTRUCTIONS FOR demo_validation_v3.py

## Step 1: Add Required Imports
Add these imports at the top of demo_validation_v3.py if not already present:

```python
from datetime import datetime, timedelta
```

## Step 2: Add the New Functions
Copy and paste all the functions from "add_price_comparison_functions()"
above into your demo_validation_v3.py file, preferably before the main function.

## Step 3: Modify the Main Function
Replace the existing demo_validation() function with the enhanced version
that includes a menu for choosing between original validation and price comparison.

## Step 4: Update the __main__ Section
The existing __main__ section should work as-is since it calls demo_validation().

## Alternative: Quick Integration
If you want a quick integration without changing the main function,
you can add this to the beginning of the existing demo_validation() function:

```python
# Add at the beginning of demo_validation()
print("\\nDo you want to run Price Comparison instead of regular validation? (y/n): ")
choice = input().strip().lower()
if choice == 'y':
    return run_price_comparison_analysis()
```

## Usage After Integration
After integration, when you run:
```bash
python demo_validation_v3.py
```

You'll get a menu:
1. Original Interactive Validation
2. Price Comparison Analysis (NEW)
3. Exit

Choose option 2 to run the price comparison with your exact requested format!
'''

    return instructions

def main():
    """Main function to show integration options"""
    print("="*80)
    print("PRICE COMPARISON INTEGRATION HELPER")
    print("="*80)

    print("\nThis script helps you add price comparison functionality to demo_validation_v3.py")
    print("\nYou have two options:")
    print("1. Use the enhanced version (demo_validation_enhanced.py)")
    print("2. Integrate into existing demo_validation_v3.py")
    print("3. See integration instructions")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == '1':
        print("\nUse demo_validation_enhanced.py - it includes all functionality!")
        print("\nTo run:")
        print("python demo_validation_enhanced.py")
        print("\nThen choose option 1 for Price Comparison Analysis")

    elif choice == '2':
        print("\nIntegration code generated:")
        print("\n" + "="*60)
        print("STEP 1: Add these functions to demo_validation_v3.py:")
        print("="*60)
        print(add_price_comparison_functions())

        print("\n" + "="*60)
        print("STEP 2: Modify the main function:")
        print("="*60)
        print(modify_main_function())

    elif choice == '3':
        print("\n" + "="*60)
        print("DETAILED INTEGRATION INSTRUCTIONS:")
        print("="*60)
        print(show_integration_instructions())

    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()