#!/usr/bin/env python3
"""
Simple test script for Nautilus Validation system
"""

import pandas as pd
from nautilus_validation.validators import ValidationEngine, ValidationRule

def test_validation():
    """Test validation engine with sample data"""
    print("Testing Validation Engine...")

    # Create sample data
    source_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'david@test.com', 'eve@test.com'],
        'age': [25, 30, 35, 28, 32]
    })

    target_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'david@test.com', 'eve@test.com'],
        'age': [25, 30, 35, 28, 32]
    })

    # Initialize validation engine
    config = {
        'row_count_validation': True,
        'data_integrity_validation': True,
        'full_data_comparison': True,
        'custom_validation_rules': True,
        'row_count_tolerance': 0,
        'null_value_tolerance': 0.01,
        'sample_size': 100
    }

    engine = ValidationEngine(config)

    # Run validation
    result = engine.validate(source_data, target_data, key_columns=['id'])

    print(f"Validation completed!")
    print(f"Overall Status: {result.overall_status.value}")
    print(f"Source Rows: {result.total_rows_source}")
    print(f"Target Rows: {result.total_rows_target}")
    print(f"Validations Run: {len(result.validation_results)}")

    for val_result in result.validation_results:
        print(f"- {val_result.validation_type}: {val_result.status.value}")

    return True

def test_requirements():
    """Test if all required packages are available"""
    print("Testing Package Dependencies...")

    required_packages = [
        'boto3', 'pyarrow', 'pandas', 'sqlalchemy',
        'pymysql', 'yaml', 'click', 'colorama',
        'jinja2', 'tqdm', 'dotenv', 'tabulate'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'yaml':
                import yaml
            elif package == 'dotenv':
                import dotenv
            elif package == 'tabulate':
                import tabulate
            else:
                __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[MISSING] {package}")
            missing_packages.append(package)

    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("All required packages are installed!")
        return True

def main():
    """Run all tests"""
    print("Nautilus Validation System Test")
    print("=" * 50)

    # Test requirements
    if not test_requirements():
        print("ERROR: Missing required packages")
        return False

    print("\n" + "=" * 50)

    # Test validation
    try:
        if test_validation():
            print("\nSUCCESS: All tests passed!")
            print("\nNext steps:")
            print("1. Set your DigitalOcean Spaces credentials:")
            print("   set SPACES_ACCESS_KEY=your_key")
            print("   set SPACES_SECRET_KEY=your_secret")
            print("2. Test connection:")
            print("   python -m nautilus_validation.cli --list-files --parquet-prefix 'your-prefix/'")
            print("3. Run validation:")
            print("   python -m nautilus_validation.cli --parquet-prefix 'data/' --table your_table --key-columns id")
            return True
        else:
            print("ERROR: Validation test failed")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)