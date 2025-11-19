# 🔍 Nautilus Validation

A comprehensive Python tool for validating Parquet data stored in DigitalOcean Spaces against data in MySQL databases.

## 🚀 Features

- **Multiple Validation Types**: Row count comparison, data integrity checks, full data comparison, and custom validation rules
- **DigitalOcean Spaces Integration**: Seamlessly read Parquet files from DigitalOcean Spaces
- **MySQL Database Support**: Connect to MySQL databases for validation
- **Flexible Configuration**: YAML-based configuration with environment variable support
- **Comprehensive Reporting**: Generate detailed reports in JSON, HTML, and CSV formats
- **CLI Interface**: Easy-to-use command-line interface with progress indicators
- **Batch Processing**: Handle large datasets efficiently with batch processing
- **Custom Validation Rules**: Define your own validation logic
- **Detailed Logging**: Configurable logging with file rotation

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- DigitalOcean Spaces credentials
- MySQL database access

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd validation_nautilus

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## ⚙️ Configuration

### 1. Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# DigitalOcean Spaces Credentials
SPACES_ACCESS_KEY=your_spaces_access_key_here
SPACES_SECRET_KEY=your_spaces_secret_key_here

# MySQL Database Credentials
DB_USERNAME=your_db_username_here
DB_PASSWORD=your_db_password_here
```

### 2. Configuration File

The `config.yaml` file contains all configuration settings:

```yaml
# DigitalOcean Spaces Configuration
digital_ocean:
  region: "nyc3"
  bucket_name: "your-bucket-name"

# MySQL Database Configuration
database:
  host: "localhost"
  port: 3306
  database_name: "your_database"

# Validation Configuration
validation:
  row_count_validation: true
  data_integrity_validation: true
  full_data_comparison: false
  batch_size: 10000
  null_value_tolerance: 0.01

# Reporting Configuration
reporting:
  output_directory: "reports"
  format: ["json", "html"]
  include_sample_data: true
```

## 🎯 Usage

### Basic Validation

```bash
# Validate Parquet files against a database table
nautilus-validate --parquet-prefix "data/2023/" --table users --key-columns id,email

# Validate specific files
nautilus-validate --parquet-files "data/file1.parquet" "data/file2.parquet" --table transactions
```

### Advanced Options

```bash
# Run specific validation types
nautilus-validate --parquet-prefix "sales/" --table sales_data \
  --validation-types row_count,data_integrity

# Custom batch size and sample size
nautilus-validate --parquet-prefix "logs/" --table events \
  --batch-size 5000 --sample-size 500

# Generate specific report formats
nautilus-validate --parquet-prefix "data/" --table customers \
  --report-formats json,html,csv

# Use custom configuration file
nautilus-validate --config custom_config.yaml --parquet-prefix "data/" --table users
```

### Utility Commands

```bash
# List available Parquet files
nautilus-validate --list-files --parquet-prefix "data/"

# Dry run to see what would be validated
nautilus-validate --dry-run --parquet-prefix "data/" --table users

# Verbose logging
nautilus-validate --verbose --parquet-prefix "data/" --table users
```

## 📊 Validation Types

### 1. Row Count Validation
Compares the number of rows between Parquet files and database tables.

### 2. Data Integrity Validation
- Null value percentage comparison
- Data type validation
- Unique value count comparison
- Column presence validation

### 3. Full Data Comparison
- Row-by-row comparison using key columns
- Identifies missing or extra rows
- Detects data value differences
- Resource intensive, use with caution

### 4. Custom Validation Rules
Define your own validation logic using Python functions.

## 📈 Reports

Nautilus Validation generates comprehensive reports in multiple formats:

### JSON Report
Structured data with all validation results and metadata.

### HTML Report
Human-readable report with:
- Summary statistics
- Detailed validation results
- Color-coded status indicators
- Sample data for issues found

### CSV Report
Tabular format suitable for spreadsheet analysis.

## 🔧 Custom Validation Rules

Create custom validation rules by defining Python functions:

```python
def no_empty_strings_validator(df):
    """Custom rule to check for empty strings"""
    return {
        'passed': not any(df.astype(str).eq('').any().values),
        'message': 'No empty strings found' if not any(df.astype(str).eq('').any().values) else 'Empty strings found'
    }

# Add to validation engine (extend the CLI or use programmatically)
from nautilus_validation.validators import ValidationRule
custom_rule = ValidationRule(
    name="no_empty_strings",
    description="Check for empty string values",
    validator=no_empty_strings_validator,
    severity="WARNING"
)
```

## 🐛 Python API

Use Nautilus Validation programmatically:

```python
from nautilus_validation import ValidationEngine, SpacesConnector, DatabaseConnector, ConfigManager

# Load configuration
config = ConfigManager('config.yaml')

# Initialize connectors
spaces = SpacesConnector(config.get_digital_ocean_config())
database = DatabaseConnector(config.get_database_config())

# Read data
parquet_files = spaces.list_parquet_files('data/2023/')
parquet_data = [spaces.read_parquet_file(f) for f in parquet_files]
db_data = list(database.read_table_data('users'))

# Combine data
import pandas as pd
source_df = pd.concat(parquet_data, ignore_index=True)
target_df = pd.concat(db_data, ignore_index=True)

# Run validation
engine = ValidationEngine(config.get_validation_config())
result = engine.validate(source_df, target_df, key_columns=['id'])

# Generate reports
from nautilus_validation.reporting import ValidationReporter
reporter = ValidationReporter()
reporter.generate_report(result)
```

## 📝 Logging

Configure logging in `config.yaml`:

```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "validation.log"
  max_file_size: "10MB"
  backup_count: 5
```

## 🚨 Error Handling

The tool includes comprehensive error handling:

- Configuration validation
- Connection error recovery
- Data reading error handling
- Graceful failure reporting
- Detailed error messages

## 🔒 Security

- Credentials stored in environment variables
- No sensitive data in configuration files
- Secure database connections
- Input validation and sanitization

## 📋 Requirements

See `requirements.txt` for complete list:

- `boto3` - DigitalOcean Spaces S3-compatible API
- `pyarrow` - Parquet file reading
- `pandas` - Data manipulation
- `sqlalchemy` - Database ORM
- `pymysql` - MySQL driver
- `pyyaml` - Configuration parsing
- `click` - CLI interface
- `colorama` - Colored terminal output
- `jinja2` - HTML report templating
- `tqdm` - Progress bars

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Connection Errors**: Verify credentials and network connectivity
2. **Memory Issues**: Reduce batch size for large datasets
3. **Configuration Errors**: Validate YAML syntax and environment variables
4. **Permission Errors**: Ensure proper file and directory permissions

### Debug Mode

Run with verbose logging for detailed debugging:

```bash
nautilus-validate --verbose --parquet-prefix "data/" --table users
```

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the log files for detailed error messages
3. Create an issue with details about your environment and configuration