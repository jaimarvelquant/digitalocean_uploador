"""
Command-line interface for Nautilus Validation
"""

import argparse
import logging
import sys
import os
from typing import List, Optional
from datetime import datetime
import colorama
from colorama import Fore, Style
from tqdm import tqdm

from .config import ConfigManager
from .connectors import SpacesConnector, DatabaseConnector
from .validators import ValidationEngine, ValidationRule, ValidationResult
from .reporting import ValidationReporter
from .utils import setup_logging, print_validation_summary

# Initialize colorama for cross-platform colored output
colorama.init()

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Nautilus Validation - Validate Parquet data against SQL databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate specific parquet files against a database table
  nautilus-validate --parquet-prefix "data/2023/" --table users --key-columns id,email

  # Validate with custom configuration
  nautilus-validate --config custom_config.yaml --parquet-prefix "sales/" --table transactions

  # Run only specific validation types
  nautilus-validate --parquet-prefix "logs/" --table events --validation-types row_count,data_integrity

  # Generate reports in specific formats
  nautilus-validate --parquet-prefix "data/" --table customers --report-formats json,html

  # List available parquet files
  nautilus-validate --list-files --parquet-prefix "data/"
        """
    )

    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (default: config.yaml)'
    )

    # Data sources
    parser.add_argument(
        '--parquet-prefix', '-p',
        type=str,
        help='Prefix for Parquet files in DigitalOcean Spaces'
    )

    parser.add_argument(
        '--table', '-t',
        type=str,
        help='Database table name to validate against'
    )

    parser.add_argument(
        '--key-columns', '-k',
        type=str,
        help='Comma-separated list of key columns for full comparison'
    )

    parser.add_argument(
        '--parquet-files',
        type=str,
        nargs='+',
        help='Specific Parquet file keys to validate (alternative to prefix)'
    )

    # Validation options
    parser.add_argument(
        '--validation-types',
        type=str,
        default='row_count,data_integrity',
        help='Comma-separated validation types: row_count,data_integrity,full_data,custom'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for processing (overrides config)'
    )

    parser.add_argument(
        '--sample-size',
        type=int,
        help='Sample size for full comparison (overrides config)'
    )

    # Reporting
    parser.add_argument(
        '--report-formats',
        type=str,
        default='json,html',
        help='Comma-separated report formats: json,html,csv'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for reports (overrides config)'
    )

    parser.add_argument(
        '--report-prefix',
        type=str,
        help='Prefix for report filenames'
    )

    # Utility options
    parser.add_argument(
        '--list-files', '-l',
        action='store_true',
        help='List available Parquet files and exit'
    )

    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be validated without executing'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output except errors'
    )

    parser.add_argument(
        '--no-reports',
        action='store_true',
        help='Skip generating reports'
    )

    return parser


def setup_argument_logging(args: argparse.Namespace, config: ConfigManager):
    """Setup logging based on arguments and config"""
    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "ERROR"
    else:
        log_level = config.get('logging.level', 'INFO')

    log_config = config.get_logging_config()
    log_config['level'] = log_level

    setup_logging(log_config)


def validate_arguments(args: argparse.Namespace, config: ConfigManager) -> bool:
    """Validate command-line arguments"""
    errors = []

    if not args.list_files and not args.parquet_prefix and not args.parquet_files:
        errors.append("Either --parquet-prefix or --parquet-files must be specified")

    if not args.list_files and not args.table:
        errors.append("--table must be specified for validation")

    if args.parquet_files and args.table:
        # Check if table exists in database
        try:
            db_config = config.get_database_config()
            db_connector = DatabaseConnector(db_config)
            table_info = db_connector.get_table_info(args.table)
            db_connector.close()
        except Exception as e:
            errors.append(f"Cannot access table '{args.table}': {e}")

    if errors:
        for error in errors:
            print(f"{Fore.RED}Error: {error}{Style.RESET_ALL}")
        return False

    return True


def list_parquet_files(spaces_connector: SpacesConnector, prefix: str):
    """List Parquet files in DigitalOcean Spaces"""
    print(f"{Fore.BLUE}Listing Parquet files with prefix: {prefix}{Style.RESET_ALL}")

    try:
        files = spaces_connector.list_parquet_files(prefix)

        if not files:
            print(f"{Fore.YELLOW}No Parquet files found with prefix: {prefix}{Style.RESET_ALL}")
            return

        print(f"{Fore.GREEN}Found {len(files)} Parquet files:{Style.RESET_ALL}")
        for file in sorted(files):
            file_info = spaces_connector.get_file_info(file)
            size_mb = file_info['size'] / (1024 * 1024)
            print(f"  {file} ({size_mb:.2f} MB)")

    except Exception as e:
        print(f"{Fore.RED}Error listing files: {e}{Style.RESET_ALL}")
        return 1


def perform_validation(args: argparse.Namespace, config: ConfigManager) -> ValidationResult:
    """Perform the actual validation"""
    print(f"{Fore.BLUE}Starting validation...{Style.RESET_ALL}")

    # Initialize connectors
    do_config = config.get_digital_ocean_config()
    db_config = config.get_database_config()

    spaces_connector = SpacesConnector(do_config)
    database_connector = DatabaseConnector(db_config)

    try:
        # Get Parquet files
        if args.parquet_files:
            parquet_files = args.parquet_files
        else:
            parquet_files = spaces_connector.list_parquet_files(args.parquet_prefix)

        if not parquet_files:
            raise ValueError(f"No Parquet files found with prefix: {args.parquet_prefix}")

        print(f"{Fore.GREEN}Found {len(parquet_files)} Parquet files{Style.RESET_ALL}")

        # Get table info
        table_info = database_connector.get_table_info(args.table)
        print(f"{Fore.GREEN}Table '{args.table}' has {table_info['row_count']:,} rows{Style.RESET_ALL}")

        # Setup validation engine
        validation_config = config.get_validation_config()
        validation_config['row_count_validation'] = 'row_count' in args.validation_types
        validation_config['data_integrity_validation'] = 'data_integrity' in args.validation_types
        validation_config['full_data_comparison'] = 'full_data' in args.validation_types

        if args.batch_size:
            validation_config['batch_size'] = args.batch_size
        if args.sample_size:
            validation_config['sample_size'] = args.sample_size

        validation_engine = ValidationEngine(validation_config)

        # Add custom validation rules (could be extended based on config)
        if 'custom' in args.validation_types:
            # Example custom rule - could be loaded from config
            def no_empty_strings_validator(df):
                return {
                    'passed': not any(df.astype(str).eq('').any().values),
                    'message': 'No empty strings found' if not any(df.astype(str).eq('').any().values) else 'Empty strings found'
                }

            custom_rule = ValidationRule(
                name="no_empty_strings",
                description="Check for empty string values",
                validator=no_empty_strings_validator,
                severity="WARNING"
            )
            validation_engine.add_custom_rule(custom_rule)

        # Read Parquet data
        print(f"{Fore.BLUE}Reading Parquet data...{Style.RESET_ALL}")
        # Don't filter columns - read all columns from Parquet files

        with tqdm(total=len(parquet_files), desc="Reading files") as pbar:
            parquet_dfs = []
            for file in parquet_files:
                try:
                    df = spaces_connector.read_parquet_file(file, columns=None)
                    parquet_dfs.append(df)
                    pbar.update(1)
                except Exception as e:
                    logger.warning(f"Error reading {file}: {e}")
                    pbar.update(1)
                    continue

        if not parquet_dfs:
            raise ValueError("No Parquet data could be read")

        # Combine all Parquet data
        import pandas as pd
        source_df = pd.concat(parquet_dfs, ignore_index=True)
        print(f"{Fore.GREEN}Read {len(source_df):,} rows from Parquet files{Style.RESET_ALL}")

        # Read database data in batches
        print(f"{Fore.BLUE}Reading database data...{Style.RESET_ALL}")
        db_dfs = []
        batch_size = validation_config.get('batch_size', 10000)
        offset = 0

        # Only select columns that exist in both datasets (exclude 'coi')
        common_columns = list(set(source_df.columns) & set(table_info['column_names']))

        with tqdm(desc="Reading database") as pbar:
            while True:
                try:
                    batch_df = next(database_connector.read_table_data(
                        args.table,
                        columns=common_columns,
                        batch_size=batch_size,
                        offset=offset
                    ))
                    if batch_df.empty:
                        break
                    db_dfs.append(batch_df)
                    offset += batch_size
                    pbar.update(len(batch_df))
                except StopIteration:
                    break

        if not db_dfs:
            raise ValueError("No database data could be read")

        target_df = pd.concat(db_dfs, ignore_index=True)
        print(f"{Fore.GREEN}Read {len(target_df):,} rows from database{Style.RESET_ALL}")

        # Perform validation
        print(f"{Fore.BLUE}Running validations...{Style.RESET_ALL}")
        key_columns = args.key_columns.split(',') if args.key_columns else None

        result = validation_engine.validate(source_df, target_df, key_columns)

        return result

    finally:
        # Clean up connections
        database_connector.close()


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    try:
        # Load configuration
        config = ConfigManager(args.config)

        # Validate configuration
        if not config.validate_config():
            print(f"{Fore.RED}Configuration validation failed. Please check your config file.{Style.RESET_ALL}")
            return 1

        # Setup logging
        setup_argument_logging(args, config)

        # Validate arguments
        if not validate_arguments(args, config):
            return 1

        # Initialize connectors
        do_config = config.get_digital_ocean_config()
        spaces_connector = SpacesConnector(do_config)

        # Handle list files option
        if args.list_files:
            prefix = args.parquet_prefix or ""
            return list_parquet_files(spaces_connector, prefix)

        # Handle dry run
        if args.dry_run:
            print(f"{Fore.YELLOW}DRY RUN - The following would be validated:{Style.RESET_ALL}")
            if args.parquet_prefix:
                files = spaces_connector.list_parquet_files(args.parquet_prefix)
                print(f"  Parquet files with prefix '{args.parquet_prefix}': {len(files)} files")
            else:
                print(f"  Specific Parquet files: {args.parquet_files}")
            print(f"  Database table: {args.table}")
            print(f"  Validation types: {args.validation_types}")
            print(f"  Report formats: {args.report_formats}")
            return 0

        # Perform validation
        result = perform_validation(args, config)

        # Print summary
        print_validation_summary(result)

        # Generate reports
        if not args.no_reports:
            print(f"{Fore.BLUE}Generating reports...{Style.RESET_ALL}")

            reporting_config = config.get_reporting_config()
            if args.output_dir:
                reporting_config['output_directory'] = args.output_dir

            reporter = ValidationReporter(
                output_dir=reporting_config['output_directory'],
                config=reporting_config
            )

            report_formats = args.report_formats.split(',')
            report_prefix = args.report_prefix

            report_files = reporter.generate_report(result, report_formats, report_prefix)

            print(f"{Fore.GREEN}Reports generated:{Style.RESET_ALL}")
            for format_type, file_path in report_files.items():
                print(f"  {format_type.upper()}: {file_path}")

        # Return appropriate exit code
        if result.overall_status.value == 'PASSED':
            return 0
        elif result.overall_status.value == 'WARNING':
            return 0  # Warnings are not failures
        else:
            return 1

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Validation interrupted by user{Style.RESET_ALL}")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return 1


if __name__ == '__main__':
    sys.exit(main())