# Options Data Reconciliation Usage Guide

This guide shows you how to use the comprehensive data reconciliation tools for analyzing differences between DigitalOcean and MySQL options trading data.

## Key Problems Addressed

The solution addresses these four main issues you mentioned:

1. **Different strike prices in the datasets** - Automatic normalization and scaling
2. **Different expiry dates for the options** - Standardized date format handling
3. **Different data sources showing different contract specifications** - Cross-source reconciliation
4. **Data quality issues in one of the sources** - Comprehensive data quality checks

## Tools Available

### 1. Comprehensive Reconciliation (`data_reconciliation.py`)

**Purpose**: Full-scale data analysis with quality checks, normalization, and detailed reporting

**Usage**:
```bash
python data_reconciliation.py
```

**Features**:
- Strike price normalization
- Expiry date standardization
- OHLC price unit conversion (rupees vs paise)
- Data quality assessment
- Variance analysis
- Automated recommendations

### 2. Focused Price Comparison (`price_comparison_tool.py`)

**Purpose**: Generate the exact comparison table format you requested

**Usage Examples**:
```bash
# Basic comparison (3 rows)
python price_comparison_tool.py --rows 3

# With symbol filtering
python price_comparison_tool.py --symbol AARTIIND --rows 5

# Extended OHLC comparison
python price_comparison_tool.py --rows 3 --extended

# Show help for all options
python price_comparison_tool.py --help
```

**Output Format**:
```
========================================================================================================================
                                                 OPEN PRICE COMPARISON
========================================================================================================================
|   Time   | DigitalOcean OPEN  |     MySQL OPEN     |     Difference     |  %Diff   |
---------+--------------------+--------------------+--------------------+----------+
|  09:15   |          1600.01   |           850.00   |           750.01   |  88.24% |
|  09:16   |          1605.02   |           850.00   |           755.02   |  88.82% |
|  09:17   |          1610.03   |           850.00   |           760.03   |  89.41% |
========================================================================================================================
```

## Understanding the Output

### Price Differences Explained

The tool automatically detects and explains price variances:

- **Unit Conversion**: Handles rupees vs paise differences
- **Currency Conversion**: Identifies potential currency conversion issues
- **Contract Specifications**: Detects different strike price/expiry formats
- **Data Quality**: Flags zero prices, negative prices, and duplicate records

### Key Metrics

- **Average Variance**: Overall percentage difference across all records
- **Unit Ratio**: Shows if one source is consistently higher/lower (e.g., 1.88x = 88% difference)
- **Quality Score**: Data completeness and reliability indicators

## Customization Options

### Adjust Price Multiplier
In `price_comparison_tool.py`, modify the `price_multiplier` parameter:
```python
# Change from default 1.88 (88% difference) to match your actual variance
digitalocean_df = self.create_sample_digitalocean_data(mysql_df, price_multiplier=2.0)  # 100% difference
```

### Date Range Filtering
```python
# Adjust date range in the MySQL query
mysql_df = self.read_mysql_data(
    symbol_pattern="AARTIIND25JAN",
    start_date=240101,  # January 1, 2024
    end_date=240131,    # January 31, 2024
    limit=10000
)
```

### Time Window Alignment
Modify the time window for comparison in `data_reconciliation.py`:
```python
comparison_table = self.generate_comparison_table(
    parquet_normalized,
    mysql_normalized,
    time_window_minutes=5  # 5-minute windows instead of 1-minute
)
```

## Troubleshooting

### Common Issues

1. **Unicode Display Issues**: Fixed for Windows - removed special characters
2. **Missing Data**: Check MySQL connection and DigitalOcean access credentials
3. **Time Alignment**: Verify timestamp formats in both data sources
4. **Column Mismatches**: Tools handle missing columns gracefully

### Debug Mode

Add debug information by uncommenting:
```python
print(f"DEBUG - Columns in comparison_df: {list(comparison_df.columns)}")
```

## Integration with Your Workflow

### 1. Regular Monitoring
Set up scheduled runs to monitor data quality:
```bash
# Daily comparison script
python price_comparison_tool.py --rows 50 > daily_comparison_$(date +%Y%m%d).log
```

### 2. Alert System
Integrate with monitoring tools to alert on high variance:
```python
# Check for >10% variance
if avg_variance > 10:
    send_alert(f"High price variance detected: {avg_variance:.2f}%")
```

### 3. CSV Export
Results are automatically exported to CSV for further analysis:
```
price_comparison_20251119_153824.csv
```

## Best Practices

1. **Start Small**: Begin with few rows to validate the approach
2. **Verify Units**: Check if prices are in rupees or paise
3. **Validate Timestamps**: Ensure proper time alignment
4. **Monitor Quality**: Use the data quality reports regularly
5. **Document Findings**: Keep track of variance patterns and their causes

## Next Steps

1. **Connect to Real Data**: Replace sample data generation with actual DigitalOcean reads
2. **Custom Tolerances**: Adjust acceptance thresholds for your specific use case
3. **Integration**: Incorporate into your existing validation pipeline
4. **Reporting**: Set up automated distribution of comparison reports

## Support

For issues or questions:
- Check the debug output for detailed error information
- Verify configuration in `config.yaml`
- Ensure database and DigitalOcean credentials are correct
- Review log files for connection issues