"""
Utility functions for Nautilus Validation
"""

import logging
import logging.handlers
import os
import sys
from typing import Dict, Any
from datetime import datetime
import colorama
from colorama import Fore, Style

from .validators import ValidationResult, ValidationStatus

colorama.init()


def setup_logging(config: Dict[str, Any]):
    """Setup logging configuration"""
    log_level = getattr(logging, config.get('level', 'INFO').upper())
    log_format = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = config.get('file', 'validation.log')
    max_file_size = config.get('max_file_size', '10MB')
    backup_count = config.get('backup_count', 5)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        # Parse file size
        if max_file_size.endswith('MB'):
            max_bytes = int(max_file_size[:-2]) * 1024 * 1024
        elif max_file_size.endswith('KB'):
            max_bytes = int(max_file_size[:-2]) * 1024
        else:
            max_bytes = int(max_file_size)

        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def print_validation_summary(result: ValidationResult):
    """Print a formatted validation summary"""
    print(f"\n{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}🔍 VALIDATION SUMMARY{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")

    # Overall status
    status_color = {
        ValidationStatus.PASSED: Fore.GREEN,
        ValidationStatus.FAILED: Fore.RED,
        ValidationStatus.WARNING: Fore.YELLOW,
        ValidationStatus.ERROR: Fore.RED
    }.get(result.overall_status, Fore.WHITE)

    print(f"\nOverall Status: {status_color}{result.overall_status.value}{Style.RESET_ALL}")
    print(f"Source Rows: {result.total_rows_source:,}")
    print(f"Target Rows: {result.total_rows_target:,}")

    if result.start_time and result.end_time:
        duration = result.end_time - result.start_time
        print(f"Duration: {duration.total_seconds():.3f} seconds")

    # Validation results
    print(f"\n{Fore.BLUE}Validation Results:{Style.RESET_ALL}")
    for val_result in result.validation_results:
        result_color = {
            ValidationStatus.PASSED: Fore.GREEN,
            ValidationStatus.FAILED: Fore.RED,
            ValidationStatus.WARNING: Fore.YELLOW,
            ValidationStatus.ERROR: Fore.RED
        }.get(val_result.status, Fore.WHITE)

        print(f"  {val_result.validation_type.replace('_', ' ').title()}: "
              f"{result_color}{val_result.status.value}{Style.RESET_ALL} "
              f"({val_result.execution_time:.3f}s)")

        if val_result.issues:
            for issue in val_result.issues[:3]:  # Show first 3 issues
                issue_color = {
                    "ERROR": Fore.RED,
                    "WARNING": Fore.YELLOW,
                    "INFO": Fore.CYAN
                }.get(issue.severity, Fore.WHITE)

                print(f"    {issue_color}• {issue.issue_type}: {issue.message}{Style.RESET_ALL}")

            if len(val_result.issues) > 3:
                print(f"    ... and {len(val_result.issues) - 3} more issues")

    # Summary statistics
    total_validations = len(result.validation_results)
    passed = len([r for r in result.validation_results if r.status == ValidationStatus.PASSED])
    failed = len([r for r in result.validation_results if r.status == ValidationStatus.FAILED])
    warnings = len([r for r in result.validation_results if r.status == ValidationStatus.WARNING])

    print(f"\n{Fore.BLUE}Summary:{Style.RESET_ALL}")
    print(f"  Total Validations: {total_validations}")
    print(f"  {Fore.GREEN}Passed: {passed}{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}Warnings: {warnings}{Style.RESET_ALL}")
    print(f"  {Fore.RED}Failed: {failed}{Style.RESET_ALL}")

    print(f"\n{Fore.BLUE}{'='*60}{Style.RESET_ALL}")


def format_bytes(bytes_value: int) -> str:
    """Format bytes in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def safe_filename(filename: str) -> str:
    """Create a safe filename by removing/replacing problematic characters"""
    import re
    # Replace problematic characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    safe_name = safe_name.strip('. ')
    # Limit length
    if len(safe_name) > 255:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:255-len(ext)] + ext
    return safe_name


def create_directory_if_not_exists(directory: str):
    """Create directory if it doesn't exist"""
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        raise PermissionError(f"Cannot create directory {directory}: {e}")


def get_dataframe_memory_usage(df) -> str:
    """Get memory usage of DataFrame in human-readable format"""
    memory_bytes = df.memory_usage(deep=True).sum()
    return format_bytes(memory_bytes)


def validate_dataframe_compatibility(df1, df2) -> Dict[str, Any]:
    """
    Validate that two DataFrames can be compared

    Returns:
        Dictionary with compatibility information
    """
    compatibility = {
        'compatible': True,
        'issues': [],
        'common_columns': list(set(df1.columns) & set(df2.columns)),
        'df1_only_columns': list(set(df1.columns) - set(df2.columns)),
        'df2_only_columns': list(set(df2.columns) - set(df1.columns))
    }

    # Check for empty DataFrames
    if df1.empty and df2.empty:
        compatibility['issues'].append("Both DataFrames are empty")
    elif df1.empty:
        compatibility['issues'].append("Source DataFrame is empty")
    elif df2.empty:
        compatibility['issues'].append("Target DataFrame is empty")

    # Check for common columns
    if not compatibility['common_columns']:
        compatibility['compatible'] = False
        compatibility['issues'].append("No common columns between DataFrames")

    return compatibility


def truncate_string(text: str, max_length: int = 50) -> str:
    """Truncate string to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def get_sample_values(series, sample_size: int = 5) -> list:
    """Get sample values from a pandas Series"""
    try:
        if series.dtype == 'object':
            # For object types, get non-null sample values
            non_null = series.dropna()
            if not non_null.empty:
                samples = non_null.sample(min(sample_size, len(non_null)), random_state=42).tolist()
                return [truncate_string(str(v)) for v in samples]
        else:
            # For numeric types, get min, max, and some random values
            samples = []
            if not series.empty:
                samples.append(f"Min: {series.min()}")
                samples.append(f"Max: {series.max()}")
                if len(series) > 2:
                    median_sample = series.median()
                    samples.append(f"Median: {median_sample}")
            return samples

        return []
    except Exception:
        return ["Unable to sample values"]