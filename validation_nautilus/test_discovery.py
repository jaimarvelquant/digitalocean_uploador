#!/usr/bin/env python3
"""
Simple test script to debug symbol discovery
"""

from nautilus_validation.config import ConfigManager
from nautilus_validation.connectors import SpacesConnector

def test_symbol_discovery():
    print("=" * 60)
    print("TESTING SYMBOL DISCOVERY")
    print("=" * 60)

    # Load config and connect
    config = ConfigManager('config.yaml')
    do_config = config.get_digital_ocean_config()
    do_config['bucket_name'] = 'historical-db-1min'
    spaces = SpacesConnector(do_config)

    # First, let's see what prefixes are available in nautilas-data/
    print("\n=== CHECKING AVAILABLE PREFIXES ===")
    prefixes_response = spaces.client.list_objects_v2(
        Bucket='historical-db-1min',
        Prefix='nautilas-data/',
        Delimiter='/',
        MaxKeys=100
    )

    prefixes = []
    if 'CommonPrefixes' in prefixes_response:
        for prefix_obj in prefixes_response['CommonPrefixes']:
            prefix_name = prefix_obj['Prefix']
            clean_prefix = prefix_name.replace('nautilas-data/', '').rstrip('/')
            if clean_prefix:
                prefixes.append(clean_prefix)
                print(f"  - {clean_prefix}")

    # Search for NIFTY across all prefixes
    print(f"\n=== SEARCHING FOR NIFTY ACROSS ALL PREFIXES ===")
    found_nifty = False

    for prefix in prefixes:
        print(f"\nSearching in: {prefix}")
        all_objects = spaces.client.list_objects_v2(
            Bucket='historical-db-1min',
            Prefix=f'nautilas-data/{prefix}/',
            MaxKeys=200
        )

        files = [obj['Key'] for obj in all_objects.get('Contents', [])]
        nifty_files = [f for f in files if 'nifty' in f.lower() or 'NIFTY' in f]

        if nifty_files:
            print(f"  Found {len(nifty_files)} NIFTY files!")
            found_nifty = True
            for file in nifty_files[:3]:  # Show first 3
                print(f"    - {file}")
        else:
            print(f"  No NIFTY files found (checked {len(files)} total files)")

    if not found_nifty:
        print(f"\n=== NO NIFTY FILES FOUND IN ANY PREFIX ===")
        print("Let's search the entire nautilas-data directory directly:")

        all_objects_full = spaces.client.list_objects_v2(
            Bucket='historical-db-1min',
            Prefix='nautilas-data/',
            MaxKeys=1000
        )

        all_files = [obj['Key'] for obj in all_objects_full.get('Contents', [])]
        all_nifty_files = [f for f in all_files if 'nifty' in f.lower() or 'NIFTY' in f]

        if all_nifty_files:
            print(f"Found {len(all_nifty_files)} NIFTY files in entire nautilas-data:")
            for file in all_nifty_files[:5]:
                print(f"  - {file}")
        else:
            print("No NIFTY files found anywhere in nautilas-data")
            print("\nChecking for related symbols:")
            related_symbols = ['BANK', 'FINNIFTY', 'SENSEX']
            for symbol in related_symbols:
                symbol_files = [f for f in all_files if symbol.lower() in f.lower()]
                if symbol_files:
                    print(f"  Found {len(symbol_files)} {symbol} files:")
                    for file in symbol_files[:2]:
                        print(f"    - {file}")

    
if __name__ == "__main__":
    test_symbol_discovery()