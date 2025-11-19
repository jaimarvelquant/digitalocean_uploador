# 🚀 Setup Guide for Nautilus Validation

## ✅ Installation Status

All dependencies are successfully installed! Your system is ready to use.

## 📋 Current Configuration

Your system is configured with:

### DigitalOcean Spaces:
- **Endpoint**: `https://blr1.digitaloceanspaces.com`
- **Region**: `blr1`
- **Bucket**: `historical-db-tick`

### MySQL Database:
- **Host**: `106.51.63.60`
- **Port**: `3306`
- **Database**: `historicaldb`
- **Username**: `mahesh`

## 🔧 Next Steps

### 1. Set DigitalOcean Credentials

You need to set your DigitalOcean Spaces credentials as environment variables:

**Windows (Command Prompt):**
```cmd
set SPACES_ACCESS_KEY=your_spaces_access_key_here
set SPACES_SECRET_KEY=your_spaces_secret_key_here
```

**Windows (PowerShell):**
```powershell
$env:SPACES_ACCESS_KEY="your_spaces_access_key_here"
$env:SPACES_SECRET_KEY="your_spaces_secret_key_here"
```

**Or create a `.env` file:**
```bash
# Copy the example file
copy .env.example .env

# Edit .env with your credentials
SPACES_ACCESS_KEY=your_spaces_access_key_here
SPACES_SECRET_KEY=your_spaces_secret_key_here
```

### 2. Test DigitalOcean Connection

List files in your Spaces bucket:

```bash
python -m nautilus_validation.cli --list-files --parquet-prefix ""
```

### 3. Start Validating Data

#### Basic Validation:
```bash
# Validate all Parquet files against a database table
python -m nautilus_validation.cli --parquet-prefix "data/" --table your_table_name --key-columns id,email

# Validate specific files
python -m nautilus_validation.cli --parquet-files "file1.parquet" "file2.parquet" --table your_table_name
```

#### Advanced Options:
```bash
# Run specific validation types only
python -m nautilus_validation.cli --parquet-prefix "data/" --table your_table_name --validation-types row_count,data_integrity

# Custom batch size for large datasets
python -m nautilus_validation.cli --parquet-prefix "data/" --table your_table_name --batch-size 5000

# Generate specific report formats
python -m nautilus_validation.cli --parquet-prefix "data/" --table your_table_name --report-formats json,html,csv

# Dry run to see what would be validated
python -m nautilus_validation.cli --dry-run --parquet-prefix "data/" --table your_table_name
```

## 📊 Understanding Validation Results

The system performs several types of validation:

1. **Row Count Validation** - Compares the number of rows
2. **Data Integrity Validation** - Checks null values, data types, and uniqueness
3. **Full Data Comparison** - Row-by-row comparison using key columns
4. **Custom Validation Rules** - User-defined validation logic

### Report Formats

- **JSON** - Structured data for programmatic use
- **HTML** - Interactive human-readable reports
- **CSV** - Tabular format for spreadsheet analysis

Reports are saved in the `reports/` directory by default.

## 🔍 Example Workflow

```bash
# 1. Test connection
python -m nautilus_validation.cli --list-files --parquet-prefix "historical-data/"

# 2. Validate a specific table
python -m nautilus_validation.cli --parquet-prefix "historical-data/2023/" --table trades --key-columns trade_id,symbol

# 3. Check the generated reports
dir reports\
```

## 🛠️ Configuration Customization

Edit `config.yaml` to customize:

```yaml
validation:
  row_count_validation: true
  data_integrity_validation: true
  full_data_comparison: false  # Resource intensive
  batch_size: 10000
  null_value_tolerance: 0.01

reporting:
  output_directory: "reports"
  format: ["json", "html"]
  include_sample_data: true
```

## 🚨 Troubleshooting

### Common Issues:

1. **"Missing required DigitalOcean config"** - Set SPACES_ACCESS_KEY and SPACES_SECRET_KEY environment variables

2. **"Cannot access table"** - Verify database connection and table exists

3. **Memory issues** - Reduce batch_size in configuration

4. **Connection timeouts** - Increase timeout value in config

### Debug Mode:
```bash
python -m nautilus_validation.cli --verbose --parquet-prefix "data/" --table your_table_name
```

## 📞 Getting Help

- Check `validation.log` for detailed error messages
- Use `--help` for command-line options
- Review `README.md` for detailed documentation

---

**Your Nautilus Validation system is ready! 🎉**