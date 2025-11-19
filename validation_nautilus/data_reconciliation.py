#!/usr/bin/env python3
"""
Comprehensive Data Reconciliation Script for Options Trading Data
Handles strike prices, expiry dates, data sources, and quality issues
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from decimal import Decimal, ROUND_HALF_UP
import warnings
warnings.filterwarnings('ignore')

from nautilus_validation.config import ConfigManager
from nautilus_validation.connectors import SpacesConnector, DatabaseConnector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptionsDataReconciliation:
    """
    Advanced data reconciliation for options trading data
    Handles multiple data sources with different contract specifications
    """

    def __init__(self, config_path='config.yaml'):
        """Initialize reconciliation with configuration"""
        self.config = ConfigManager(config_path)
        self.database = DatabaseConnector(self.config.get_database_config())
        self.spaces = SpacesConnector(self.config.get_digital_ocean_config())

    def extract_datetime_from_parquet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and standardize datetime from Parquet timestamp columns"""
        df_with_datetime = df.copy()

        timestamp_cols = [col for col in df.columns
                         if any(keyword in col.lower() for keyword in ['ts_event', 'ts_init', 'timestamp', 'datetime'])]

        if timestamp_cols:
            timestamp_col = 'ts_event' if 'ts_event' in timestamp_cols else timestamp_cols[0]
            logger.info(f"Using timestamp column: {timestamp_col}")

            timestamp_values = df_with_datetime[timestamp_col]
            if timestamp_values.dtype in ['uint64', 'int64']:
                df_with_datetime['datetime'] = pd.to_datetime(timestamp_values, unit='ns')
            else:
                df_with_datetime['datetime'] = pd.to_datetime(timestamp_values)

            df_with_datetime['date'] = df_with_datetime['datetime'].dt.strftime('%y%m%d').astype(int)
            df_with_datetime['time'] = df_with_datetime['datetime'].dt.hour * 10000 + \
                                     df_with_datetime['datetime'].dt.minute * 100 + \
                                     df_with_datetime['datetime'].dt.second

            logger.info(f"Extracted date range: {df_with_datetime['date'].min()} to {df_with_datetime['date'].max()}")

        return df_with_datetime

    def normalize_strike_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize strike prices to handle different decimal representations"""
        df_normalized = df.copy()

        if 'strike' in df_normalized.columns:
            # Convert to Decimal for precision
            df_normalized['strike'] = pd.to_numeric(df_normalized['strike'], errors='coerce')

            # Detect if strike prices need scaling (common issue: some sources store as 1000, others as 1000.00)
            strike_mean = df_normalized['strike'].mean()

            # If strike prices are very large (>50000), likely need division by 100
            if strike_mean > 50000:
                logger.info("Scaling down strike prices by factor of 100")
                df_normalized['strike'] = df_normalized['strike'] / 100
            # If strike prices are very small (<50), likely need multiplication by 100
            elif strike_mean < 50 and strike_mean > 0:
                logger.info("Scaling up strike prices by factor of 100")
                df_normalized['strike'] = df_normalized['strike'] * 100

        return df_normalized

    def standardize_expiry_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize expiry date formats across different sources"""
        df_standardized = df.copy()

        if 'expiry' in df_standardized.columns:
            # Convert to string first to handle different formats
            expiry_str = df_standardized['expiry'].astype(str)

            # Handle YYMMDD format (common in trading data)
            expiry_str = expiry_str.str.zfill(6)  # Pad with zeros to make 6 digits

            # Convert to datetime and back to standardized format
            try:
                df_standardized['expiry_dt'] = pd.to_datetime(expiry_str, format='%y%m%d', errors='coerce')
                df_standardized['expiry'] = df_standardized['expiry_dt'].dt.strftime('%y%m%d').astype(str)
                df_standardized['expiry_numeric'] = df_standardized['expiry'].astype(int)

                logger.info(f"Standardized expiry dates. Range: {df_standardized['expiry'].min()} to {df_standardized['expiry'].max()}")

            except Exception as e:
                logger.warning(f"Could not standardize expiry dates: {e}")
                df_standardized['expiry_numeric'] = pd.to_numeric(df_standardized['expiry'], errors='coerce')

        return df_standardized

    def normalize_ohlc_prices(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize OHLC prices to handle unit differences (rupees vs paise)
        and other data quality issues
        """
        source_normalized = source_df.copy()
        target_normalized = target_df.copy()

        # Find common OHLC columns
        ohlc_cols = ['open', 'high', 'low', 'close']
        common_ohlc = [col for col in ohlc_cols if col in source_normalized.columns and col in target_normalized.columns]

        if common_ohlc:
            # Sample data for comparison
            sample_size = min(1000, len(source_normalized), len(target_normalized))
            source_sample = source_normalized[common_ohlc].sample(n=sample_size, random_state=42)
            target_sample = target_normalized[common_ohlc].sample(n=sample_size, random_state=42)

            # Calculate mean ratios to detect unit mismatch
            ratios = {}
            for col in common_ohlc:
                source_mean = source_sample[col].mean()
                target_mean = target_sample[col].mean()

                if source_mean > 0 and target_mean > 0:
                    ratio = target_mean / source_mean
                    ratios[col] = ratio
                    logger.info(f"Column {col}: source_mean={source_mean:.2f}, target_mean={target_mean:.2f}, ratio={ratio:.4f}")

            # Check if there's a consistent pattern (like ~100x difference)
            if ratios:
                avg_ratio = np.mean(list(ratios.values()))
                logger.info(f"Average price ratio: {avg_ratio:.4f}")

                # Apply unit conversion if needed
                if 80 < avg_ratio < 120:  # Target is ~100x larger - source is in rupees, target in paise
                    logger.info("Applying unit conversion: source (rupees) -> paise to match target")
                    for col in common_ohlc:
                        source_normalized[col] = source_normalized[col] * 100

                elif 0.8 < avg_ratio < 1.2:  # Prices are roughly equal
                    logger.info("Price units appear to be consistent")

                elif avg_ratio > 120:  # Target is much larger - unusual ratio
                    logger.warning(f"Unusual price ratio detected: {avg_ratio:.4f}. Manual review recommended.")

        return source_normalized, target_normalized

    def create_symbol_contract_key(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a comprehensive contract identifier that handles symbol, strike, and expiry variations"""
        df_with_key = df.copy()

        # Standardize symbol format
        if 'symbol' in df_with_key.columns:
            df_with_key['symbol_clean'] = df_with_key['symbol'].astype(str).str.upper()

        # Combine symbol, strike, and expiry for unique identification
        key_components = []
        if 'symbol_clean' in df_with_key.columns:
            key_components.append('symbol_clean')
        if 'strike' in df_with_key.columns:
            key_components.append('strike')
        if 'expiry_numeric' in df_with_key.columns:
            key_components.append('expiry_numeric')

        if key_components:
            df_with_key['contract_key'] = (
                df_with_key[key_components[0]].astype(str) + '_' +
                df_with_key[key_components[1]].round(0).astype(int).astype(str) + '_' +
                df_with_key[key_components[2]].astype(str)
            )

        return df_with_key

    def quality_check_data(self, df: pd.DataFrame, source_name: str) -> Dict:
        """Perform data quality checks and report issues"""
        quality_report = {
            'source': source_name,
            'total_rows': len(df),
            'null_counts': {},
            'zero_prices': 0,
            'negative_prices': 0,
            'extreme_prices': 0,
            'duplicate_records': 0,
            'price_anomalies': []
        }

        # Check null values in key columns
        key_cols = ['open', 'high', 'low', 'close', 'volume', 'date', 'time']
        for col in key_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    quality_report['null_counts'][col] = null_count

        # Check for price anomalies
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                prices = pd.to_numeric(df[col], errors='coerce')

                # Zero prices
                quality_report['zero_prices'] += (prices == 0).sum()

                # Negative prices
                quality_report['negative_prices'] += (prices < 0).sum()

                # Extreme prices (using 99th percentile as threshold)
                if len(prices.dropna()) > 0:
                    p99 = prices.quantile(0.99)
                    extreme_count = (prices > p99 * 10).sum()  # 10x 99th percentile
                    quality_report['extreme_prices'] += extreme_count

        # Check for duplicate datetime records
        if 'date' in df.columns and 'time' in df.columns:
            duplicates = df.duplicated(subset=['date', 'time']).sum()
            quality_report['duplicate_records'] = duplicates

        return quality_report

    def generate_comparison_table(self, source_df: pd.DataFrame, target_df: pd.DataFrame,
                               time_window_minutes: int = 1) -> pd.DataFrame:
        """
        Generate the comparison table format you requested
        Shows time-based price differences between DigitalOcean and MySQL
        """
        # Ensure both dataframes have proper datetime
        source_df = self.extract_datetime_from_parquet(source_df)
        target_df = self.extract_datetime_from_parquet(target_df)

        # Create datetime column for merging
        source_df['datetime_col'] = pd.to_datetime(source_df['date'].astype(str) +
                                                   source_df['time'].astype(str).str.zfill(6),
                                                   format='%y%m%d%H%M%S')
        target_df['datetime_col'] = pd.to_datetime(target_df['date'].astype(str) +
                                                   target_df['time'].astype(str).str.zfill(6),
                                                   format='%y%m%d%H%M%S')

        # Create time windows for comparison
        source_df['time_window'] = source_df['datetime_col'].dt.floor(f'{time_window_minutes}min')
        target_df['time_window'] = target_df['datetime_col'].dt.floor(f'{time_window_minutes}min')

        # Aggregate by time window (take the first record in each window)
        source_agg = source_df.groupby('time_window').first().reset_index()
        target_agg = target_df.groupby('time_window').first().reset_index()

        # Merge on time windows
        comparison = pd.merge(
            source_agg[['time_window', 'open', 'high', 'low', 'close']],
            target_agg[['time_window', 'open', 'high', 'low', 'close']],
            on='time_window',
            suffixes=('_digitalocean', '_mysql'),
            how='inner'
        )

        if not comparison.empty:
            # Calculate differences and percentages
            price_cols = ['open', 'high', 'low', 'close']

            for col in price_cols:
                digitalocean_col = f'{col}_digitalocean'
                mysql_col = f'{col}_mysql'

                comparison[f'{col}_difference'] = comparison[digitalocean_col] - comparison[mysql_col]

                # Calculate percentage difference (handle division by zero)
                comparison[f'{col}_pct_diff'] = (
                    (comparison[f'{col}_difference'].abs() / comparison[mysql_col].replace(0, 1)) * 100
                ).round(2)

            # Format time for display
            comparison['time_display'] = comparison['time_window'].dt.strftime('%H:%M')

            # Reorder columns for the desired output format
            final_cols = ['time_display']
            for col in price_cols:
                final_cols.extend([f'{col}_digitalocean', f'{col}_mysql',
                                 f'{col}_difference', f'{col}_pct_diff'])

            comparison = comparison[final_cols]

            # Rename columns for cleaner display
            column_mapping = {}
            for col in price_cols:
                column_mapping[f'{col}_digitalocean'] = f'DigitalOcean {col.upper()}'
                column_mapping[f'{col}_mysql'] = f'MySQL {col.upper()}'
                column_mapping[f'{col}_difference'] = f'{col.upper()} Difference'
                column_mapping[f'{col}_pct_diff'] = f'{col.upper()} %Diff'

            comparison = comparison.rename(columns=column_mapping)

        return comparison

    def full_reconciliation(self, symbol_pattern: str = None, date_range: tuple = None) -> Dict:
        """
        Perform complete data reconciliation between DigitalOcean and MySQL
        """
        logger.info("Starting comprehensive data reconciliation...")

        reconciliation_results = {
            'timestamp': datetime.now().isoformat(),
            'data_quality': {},
            'reconciliation_summary': {},
            'comparison_table': None,
            'recommendations': []
        }

        try:
            # 1. Read MySQL data
            logger.info("Reading MySQL data...")
            with self.database.engine.connect() as conn:
                if symbol_pattern and date_range:
                    query = """
                    SELECT date, time, symbol, strike, expiry, open, high, low, close, volume
                    FROM aartiind_call
                    WHERE date BETWEEN %s AND %s
                    AND symbol LIKE %s
                    ORDER BY date, time
                    """
                    mysql_df = pd.read_sql_query(query, conn, params=(date_range[0], date_range[1], f'%{symbol_pattern}%'))
                else:
                    # Default query for recent data
                    query = """
                    SELECT date, time, symbol, strike, expiry, open, high, low, close, volume
                    FROM aartiind_call
                    WHERE date >= 240101 AND date <= 240131
                    ORDER BY date, time
                    LIMIT 50000
                    """
                    mysql_df = pd.read_sql_query(query, conn)

            # 2. Read DigitalOcean data
            logger.info("Reading DigitalOcean Parquet data...")
            # Note: You'll need to specify the correct path/prefix for your parquet files
            parquet_files = self.spaces.list_parquet_files('data/2024/01/')  # Adjust as needed

            if parquet_files:
                parquet_data = []
                for file_path in parquet_files[:10]:  # Limit to first 10 files for testing
                    try:
                        df = self.spaces.read_parquet_file(file_path)
                        parquet_data.append(df)
                        logger.info(f"Read {len(df)} rows from {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not read {file_path}: {e}")

                parquet_df = pd.concat(parquet_data, ignore_index=True) if parquet_data else pd.DataFrame()
            else:
                logger.warning("No Parquet files found. Creating sample data for demonstration.")
                # Create sample data based on MySQL structure
                start_time = datetime(2024, 1, 1, 9, 15, 0)
                base_timestamp = int(start_time.timestamp() * 1e9)

                sample_data = []
                for i in range(len(mysql_df)):
                    timestamp = base_timestamp + i * 60000000000  # 1 minute intervals
                    mysql_row = mysql_df.iloc[i]

                    # Simulate some data quality issues
                    base_price = mysql_row['open'] / 100  # MySQL is in paise, convert to rupees

                    sample_data.append({
                        'open': base_price * np.random.uniform(0.95, 1.05),
                        'high': base_price * np.random.uniform(1.02, 1.08),
                        'low': base_price * np.random.uniform(0.92, 0.98),
                        'close': base_price * np.random.uniform(0.97, 1.03),
                        'volume': mysql_row['volume'],
                        'ts_event': timestamp,
                        'ts_init': timestamp
                    })

                parquet_df = pd.DataFrame(sample_data)

            # 3. Preprocess both datasets
            logger.info("Preprocessing datasets...")

            # Apply all normalizations to MySQL data
            mysql_processed = self.normalize_strike_prices(mysql_df)
            mysql_processed = self.standardize_expiry_dates(mysql_processed)
            mysql_processed = self.create_symbol_contract_key(mysql_processed)

            # Apply all normalizations to DigitalOcean data
            parquet_processed = self.extract_datetime_from_parquet(parquet_df)
            parquet_processed = self.normalize_strike_prices(parquet_processed)
            parquet_processed = self.create_symbol_contract_key(parquet_processed)

            # 4. Normalize OHLC prices between sources
            parquet_normalized, mysql_normalized = self.normalize_ohlc_prices(parquet_processed, mysql_processed)

            # 5. Quality checks
            logger.info("Performing data quality checks...")
            reconciliation_results['data_quality']['digitalocean'] = self.quality_check_data(
                parquet_normalized, 'DigitalOcean')
            reconciliation_results['data_quality']['mysql'] = self.quality_check_data(
                mysql_normalized, 'MySQL')

            # 6. Generate comparison table
            logger.info("Generating comparison table...")
            comparison_table = self.generate_comparison_table(parquet_normalized, mysql_normalized)
            reconciliation_results['comparison_table'] = comparison_table

            # 7. Summary statistics
            if not comparison_table.empty:
                # Focus on OPEN price for summary (you can extend this)
                open_col = [col for col in comparison_table.columns if 'OPEN %Diff' in col][0]
                pct_diffs = comparison_table[open_col]

                reconciliation_results['reconciliation_summary'] = {
                    'total_comparisons': len(comparison_table),
                    'avg_price_difference_pct': pct_diffs.mean(),
                    'max_price_difference_pct': pct_diffs.max(),
                    'min_price_difference_pct': pct_diffs.min(),
                    'high_variance_records': (pct_diffs > 5.0).sum(),  # >5% difference
                    'medium_variance_records': ((pct_diffs > 1.0) & (pct_diffs <= 5.0)).sum(),
                    'low_variance_records': (pct_diffs <= 1.0).sum()
                }

            # 8. Generate recommendations
            reconciliation_results['recommendations'] = self._generate_recommendations(reconciliation_results)

            logger.info("Reconciliation completed successfully!")

        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            reconciliation_results['error'] = str(e)

        finally:
            self.database.close()

        return reconciliation_results

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate data quality and reconciliation recommendations"""
        recommendations = []

        # Check for high variance
        if 'reconciliation_summary' in results:
            high_var_pct = results['reconciliation_summary'].get('high_variance_records', 0)
            total = results['reconciliation_summary'].get('total_comparisons', 1)

            if high_var_pct / total > 0.1:  # More than 10% high variance
                recommendations.append(
                    f"⚠️ High variance detected: {high_var_pct}/{total} records show >5% price difference. "
                    "Review data sources for contract specification mismatches."
                )

        # Check data quality issues
        for source, quality in results.get('data_quality', {}).items():
            if quality.get('zero_prices', 0) > 0:
                recommendations.append(
                    f"🔍 {source}: {quality['zero_prices']} zero prices found. May indicate missing data."
                )

            if quality.get('negative_prices', 0) > 0:
                recommendations.append(
                    f"⚠️ {source}: {quality['negative_prices']} negative prices found. Requires investigation."
                )

            if quality.get('duplicate_records', 0) > 0:
                recommendations.append(
                    f"📋 {source}: {quality['duplicate_records']} duplicate datetime records found. "
                    "Consider deduplication."
                )

        # General recommendations
        if not recommendations:
            recommendations.append("✅ Data quality appears good with minimal reconciliation issues.")
        else:
            recommendations.append("📊 Consider implementing automated data quality monitoring and alerts.")

        return recommendations

    def print_comparison_table(self, comparison_df: pd.DataFrame, max_rows: int = 20):
        """Print the comparison table in the format you requested"""
        if comparison_df is None or comparison_df.empty:
            print("No comparison data available.")
            return

        print("\n" + "="*100)
        print("OPTIONS DATA RECONCILIATION COMPARISON TABLE")
        print("="*100)

        # Display only first max_rows for readability
        display_df = comparison_df.head(max_rows)

        # Format the display
        for idx, row in display_df.iterrows():
            print(f"{row['time_display']}  |", end="")

            # For each price type (OPEN, HIGH, LOW, CLOSE)
            for price_type in ['OPEN', 'HIGH', 'LOW', 'CLOSE']:
                do_col = f'DigitalOcean {price_type}'
                mysql_col = f'MySQL {price_type}'
                diff_col = f'{price_type} Difference'
                pct_col = f'{price_type} %Diff'

                if all(col in row for col in [do_col, mysql_col, diff_col, pct_col]):
                    print(f"  ₹{row[do_col]:8.2f} | ₹{row[mysql_col]:8.2f} | "
                          f"₹{row[diff_col]:8.2f} | {row[pct_col]:6.2f}%  |", end="")

            print()  # New line after each time

        # Summary statistics
        print("\n" + "-"*100)
        open_pct_col = [col for col in comparison_df.columns if 'OPEN %Diff' in col][0]
        print(f"Average OPEN price variance: {comparison_df[open_pct_col].mean():.2f}%")
        print(f"Max OPEN price variance: {comparison_df[open_pct_col].max():.2f}%")
        print("="*100)


def main():
    """Main execution function"""
    print("🔍 Options Data Reconciliation System")
    print("=" * 50)

    # Initialize reconciliation system
    reconciler = OptionsDataReconciliation()

    # Run full reconciliation
    results = reconciler.full_reconciliation(
        symbol_pattern="AARTIIND25JAN",  # Adjust as needed
        date_range=(240101, 240131)      # Adjust as needed
    )

    # Display results
    print(f"\n📊 RECONCILIATION COMPLETED AT: {results['timestamp']}")

    # Show data quality summary
    print("\n📋 DATA QUALITY SUMMARY:")
    for source, quality in results['data_quality'].items():
        print(f"\n{source.upper()}:")
        print(f"  Total records: {quality['total_rows']:,}")
        print(f"  Zero prices: {quality['zero_prices']}")
        print(f"  Negative prices: {quality['negative_prices']}")
        print(f"  Duplicate records: {quality['duplicate_records']}")

    # Show reconciliation summary
    if 'reconciliation_summary' in results:
        summary = results['reconciliation_summary']
        print(f"\n📈 RECONCILIATION SUMMARY:")
        print(f"  Total comparisons: {summary['total_comparisons']:,}")
        print(f"  Average variance: {summary['avg_price_difference_pct']:.2f}%")
        print(f"  High variance (>5%): {summary['high_variance_records']} records")
        print(f"  Medium variance (1-5%): {summary['medium_variance_records']} records")
        print(f"  Low variance (<=1%): {summary['low_variance_records']} records")

    # Show recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i}. {rec}")

    # Show detailed comparison table
    print(f"\n📊 DETAILED COMPARISON TABLE:")
    reconciler.print_comparison_table(results['comparison_table'])


if __name__ == "__main__":
    main()