#!/usr/bin/env python3
"""
Comprehensive search for NIFTY data specifically in bar/ prefix
"""

from nautilus_validation.config import ConfigManager
from nautilus_validation.connectors import SpacesConnector

def search_bar_for_nifty():
    print("=" * 60)
    print("COMPREHENSIVE SEARCH FOR NIFTY IN bar/ PREFIX")
    print("=" * 60)

    config = ConfigManager('config.yaml')
    do_config = config.get_digital_ocean_config()
    do_config['bucket_name'] = 'historical-db-1min'
    spaces = SpacesConnector(do_config)

    prefix = 'nautilas-data/bar/'
    print(f"Searching: {prefix}")
    print("=" * 60)

    # Search with very high limit to get all files
    all_files = []
    continuation_token = None
    total_files = 0

    print("Fetching all files from bar/ prefix...")

    while True:
        kwargs = {
            'Bucket': 'historical-db-1min',
            'Prefix': prefix,
            'MaxKeys': 1000
        }

        if continuation_token:
            kwargs['ContinuationToken'] = continuation_token

        try:
            response = spaces.client.list_objects_v2(**kwargs)
            files = [obj['Key'] for obj in response.get('Contents', [])]
            total_files += len(files)
            all_files.extend(files)

            print(f"  Fetched {len(files)} files (total so far: {total_files})")

            # Check if there are more files
            if not response.get('IsTruncated', False):
                break
            continuation_token = response.get('NextContinuationToken')

        except Exception as e:
            print(f"Error fetching files: {e}")
            break

    print(f"\nTotal files found in bar/: {total_files}")

    # Search for NIFTY with multiple approaches
    print("\n=== SEARCHING FOR NIFTY (Multiple Approaches) ===")

    # Approach 1: Case-insensitive search for "nifty"
    nifty_files_case_insensitive = [f for f in all_files if 'nifty' in f.lower()]
    print(f"\n1. Case-insensitive 'nifty' search: {len(nifty_files_case_insensitive)} files")
    for f in nifty_files_case_insensitive[:5]:
        print(f"   - {f}")

    # Approach 2: Case-sensitive search for "NIFTY"
    nifty_files_case_sensitive = [f for f in all_files if 'NIFTY' in f]
    print(f"\n2. Case-sensitive 'NIFTY' search: {len(nifty_files_case_sensitive)} files")
    for f in nifty_files_case_sensitive[:5]:
        print(f"   - {f}")

    # Approach 3: Search for common NIFTY variations
    nifty_variations = [
        'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'NIFTY50', 'CNXNIFTY',
        'nifty', 'banknifty', 'finnifty', 'nifty50', 'cnxnifty'
    ]

    all_nifty_files = []
    for variation in nifty_variations:
        matching_files = [f for f in all_files if variation in f]
        if matching_files:
            print(f"\n3. Found {len(matching_files)} files with '{variation}':")
            all_nifty_files.extend(matching_files)
            for f in matching_files[:3]:
                print(f"   - {f}")

    # Remove duplicates
    unique_nifty_files = list(set(all_nifty_files))
    print(f"\n4. Total unique NIFTY-related files: {len(unique_nifty_files)}")

    # Approach 4: Look for directory names that might contain NIFTY
    print(f"\n=== SEARCHING FOR NIFTY IN DIRECTORY NAMES ===")

    directories = set()
    for file in all_files:
        path_parts = file.split('/')
        for part in path_parts:
            if part not in ['nautilas-data', 'bar'] and len(part) > 2:
                directories.add(part)

    nifty_directories = [d for d in directories if 'nifty' in d.lower() or 'NIFTY' in d]
    print(f"Found {len(nifty_directories)} directories with NIFTY:")
    for d in nifty_directories:
        print(f"  - {d}")

    # Show all unique directory names in bar/
    print(f"\n=== ALL DIRECTORY NAMES IN bar/ PREFIX ===")
    for d in sorted(directories):
        print(f"  - {d}")

    # If we found NIFTY files, analyze them
    if unique_nifty_files:
        print(f"\n=== DETAILED ANALYSIS OF NIFTY FILES ===")
        for i, file in enumerate(unique_nifty_files[:10], 1):
            print(f"\n{i}. {file}")
            path_parts = file.split('/')
            print(f"   Path parts: {path_parts}")

            # Try to extract symbol using our logic
            symbol = None
            for part in path_parts:
                part_upper = part.upper()
                if any(keyword in part_upper for keyword in ['NSE', 'BSE', 'FUTURE', 'INDEX']):
                    if 'NIFTY' in part_upper or 'BANKNIFTY' in part_upper or 'FINNIFTY' in part_upper:
                        symbol = part_upper
                        print(f"   -> Extracted symbol: {symbol}")
                        break

            if not symbol:
                print(f"   -> Could not extract symbol")
    else:
        print(f"\n=== NO NIFTY FILES FOUND ===")
        print("Let's check a sample of all files to see what's actually there:")

        parquet_files = [f for f in all_files if '.parquet' in f.lower()]
        print(f"\nSample of 20 parquet files from bar/:")

        for i, file in enumerate(parquet_files[:20], 1):
            print(f"{i:2d}. {file}")

            # Extract the symbol part
            path_parts = file.split('/')
            if len(path_parts) >= 4:  # Should have: nautilas-data/bar/symbol/filename
                symbol_part = path_parts[2]  # The third part should be the symbol
                print(f"     -> Symbol part: {symbol_part}")

if __name__ == "__main__":
    search_bar_for_nifty()