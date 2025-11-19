#!/usr/bin/env python3
"""
Test script to demonstrate the Nautilus Validation system functionality
"""

import os
import sys
from nautilus_validation.config import ConfigManager
from nautilus_validation.validators import ValidationEngine, ValidationRule
from nautilus_validation.connectors import DatabaseConnector, SpacesConnector
import pandas as pd

def test_configuration():
    """Test configuration loading"""
    print("Testing Configuration System...")

    try:
        config = ConfigManager('config.yaml')

        print(f"[OK] DigitalOcean Config:")
        do_config = config.get_digital_ocean_config()
        print(f"   Endpoint: {do_config.get('endpoint')}")
        print(f"   Region: {do_config.get('region')}")
        print(f"   Bucket: {do_config.get('bucket_name')}")

        print(f"[OK] Database Config:")
        db_config = config.get_database_config()
        print(f"   Host: {db_config.get('host')}")
        print(f"   Database: {db_config.get('database_name')}")
        print(f"   Username: {db_config.get('username')}")

        print(f"[OK] Validation Config:")
        val_config = config.get_validation_config()
        print(f"   Row Count Validation: {val_config.get('row_count_validation')}")
        print(f"   Data Integrity Validation: {val_config.get('data_integrity_validation')}")
        print(f"   Batch Size: {val_config.get('batch_size')}")

        return True
    except Exception as e:
        print(f"[ERROR] Configuration test failed: {e}")
        return False

def test_validation_engine():
    """Test validation engine with sample data"""
    print("\n🧪 Testing Validation Engine...")

    try:
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

        # Add custom validation rule
        def email_format_validator(df):
            """Custom validator to check email format"""
            if 'email' in df.columns:
                valid_emails = df['email'].str.contains('@').all()
                return {
                    'passed': valid_emails,
                    'message': 'All emails have valid format' if valid_emails else 'Invalid email format found'
                }
            return {'passed': True, 'message': 'No email column found'}

        email_rule = ValidationRule(
            name="email_format",
            description="Check email format validation",
            validator=email_format_validator,
            severity="WARNING"
        )
        engine.add_custom_rule(email_rule)

        # Run validation
        result = engine.validate(source_data, target_data, key_columns=['id'])

        print(f"✅ Validation completed!")
        print(f"   Overall Status: {result.overall_status.value}")
        print(f"   Source Rows: {result.total_rows_source}")
        print(f"   Target Rows: {result.total_rows_target}")
        print(f"   Validations Run: {len(result.validation_results)}")

        for val_result in result.validation_results:
            print(f"   - {val_result.validation_type}: {val_result.status.value}")

        return True

    except Exception as e:
        print(f"❌ Validation engine test failed: {e}")
        return False

def test_requirements():
    """Test if all required packages are available"""
    print("📦 Testing Package Dependencies...")

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
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print(f"✅ All required packages are installed!")
        return True

def main():
    """Run all tests"""
    print("🚀 Nautilus Validation System Test")
    print("=" * 50)

    tests = [
        ("Package Dependencies", test_requirements),
        ("Configuration System", test_configuration),
        ("Validation Engine", test_validation_engine),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print(f"{'='*50}")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1

    print(f"\nOverall Result: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n📖 Next Steps:")
        print("1. Set up your DigitalOcean Spaces credentials in environment variables")
        print("2. Test the connection: python -m nautilus_validation.cli --list-files --parquet-prefix 'your-prefix/'")
        print("3. Run validation: python -m nautilus_validation.cli --parquet-prefix 'data/' --table your_table --key-columns id")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)