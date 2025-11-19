"""
Validation engine and validators for Nautilus Validation
"""

import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation result status"""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


@dataclass
class ValidationRule:
    """Custom validation rule definition"""
    name: str
    description: str
    validator: Callable[[pd.DataFrame], Dict[str, Any]]
    severity: str = "ERROR"  # ERROR, WARNING, INFO


@dataclass
class ValidationIssue:
    """Individual validation issue"""
    column: str
    issue_type: str
    message: str
    count: int = 0
    severity: str = "ERROR"
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class ValidationResultSet:
    """Results of a single validation check"""
    validation_type: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[ValidationIssue] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationResult:
    """Complete validation results for a dataset comparison"""
    source: str
    target: str
    overall_status: ValidationStatus
    total_rows_source: int = 0
    total_rows_target: int = 0
    validation_results: List[ValidationResultSet] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: ValidationResultSet):
        """Add a validation result set"""
        self.validation_results.append(result)

    def get_failed_validations(self) -> List[ValidationResultSet]:
        """Get all failed validations"""
        return [r for r in self.validation_results if r.status == ValidationStatus.FAILED]

    def get_warnings(self) -> List[ValidationResultSet]:
        """Get all validation warnings"""
        return [r for r in self.validation_results if r.status == ValidationStatus.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary"""
        return {
            'source': self.source,
            'target': self.target,
            'overall_status': self.overall_status.value,
            'total_rows_source': self.total_rows_source,
            'total_rows_target': self.total_rows_target,
            'validation_results': [
                {
                    'validation_type': r.validation_type,
                    'status': r.status.value,
                    'message': r.message,
                    'details': r.details,
                    'issues': [
                        {
                            'column': i.column,
                            'issue_type': i.issue_type,
                            'message': i.message,
                            'count': i.count,
                            'severity': i.severity,
                            'sample_values': i.sample_values
                        }
                        for i in r.issues
                    ],
                    'execution_time': r.execution_time,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.validation_results
            ],
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'metadata': self.metadata,
            'summary': {
                'total_validations': len(self.validation_results),
                'passed': len([r for r in self.validation_results if r.status == ValidationStatus.PASSED]),
                'failed': len([r for r in self.validation_results if r.status == ValidationStatus.FAILED]),
                'warnings': len([r for r in self.validation_results if r.status == ValidationStatus.WARNING]),
                'errors': len([r for r in self.validation_results if r.status == ValidationStatus.ERROR])
            }
        }


class RowCountValidator:
    """Validator for row count comparison"""

    def __init__(self, tolerance: int = 0):
        """
        Initialize row count validator

        Args:
            tolerance: Allowable difference in row counts
        """
        self.tolerance = tolerance

    def validate(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> ValidationResultSet:
        """
        Validate row counts between DataFrames

        Args:
            source_df: Source DataFrame (Parquet data)
            target_df: Target DataFrame (SQL data)

        Returns:
            Validation result set
        """
        start_time = datetime.now()
        source_count = len(source_df)
        target_count = len(target_df)
        difference = abs(source_count - target_count)

        if difference <= self.tolerance:
            status = ValidationStatus.PASSED
            message = f"Row counts match: {source_count} source, {target_count} target"
        else:
            status = ValidationStatus.FAILED
            message = f"Row count mismatch: {source_count} source, {target_count} target (difference: {difference})"

        execution_time = (datetime.now() - start_time).total_seconds()

        return ValidationResultSet(
            validation_type="row_count",
            status=status,
            message=message,
            details={
                'source_count': source_count,
                'target_count': target_count,
                'difference': difference,
                'tolerance': self.tolerance,
                'percentage_difference': (difference / max(source_count, target_count, 1)) * 100
            },
            execution_time=execution_time,
            timestamp=start_time
        )


class DataIntegrityValidator:
    """Validator for data integrity checks"""

    def __init__(self, null_tolerance: float = 0.01):
        """
        Initialize data integrity validator

        Args:
            null_tolerance: Allowable percentage difference in null values (0-1)
        """
        self.null_tolerance = null_tolerance

    def validate(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> ValidationResultSet:
        """
        Validate data integrity between DataFrames

        Args:
            source_df: Source DataFrame (Parquet data)
            target_df: Target DataFrame (SQL data)

        Returns:
            Validation result set
        """
        start_time = datetime.now()
        issues = []
        details = {}

        # Get common columns
        common_columns = set(source_df.columns) & set(target_df.columns)

        if not common_columns:
            return ValidationResultSet(
                validation_type="data_integrity",
                status=ValidationStatus.FAILED,
                message="No common columns found between datasets",
                execution_time=(datetime.now() - start_time).total_seconds()
            )

        for column in common_columns:
            column_issues = []

            # Null value comparison
            source_nulls = source_df[column].isnull().sum()
            target_nulls = target_df[column].isnull().sum()
            source_total = len(source_df)
            target_total = len(target_df)

            source_null_pct = source_nulls / source_total if source_total > 0 else 0
            target_null_pct = target_nulls / target_total if target_total > 0 else 0
            null_diff_pct = abs(source_null_pct - target_null_pct)

            if null_diff_pct > self.null_tolerance:
                column_issues.append(ValidationIssue(
                    column=column,
                    issue_type="null_mismatch",
                    message=f"Null value percentage difference exceeds tolerance: {null_diff_pct:.2%} vs {self.null_tolerance:.2%}",
                    count=abs(source_nulls - target_nulls),
                    severity="WARNING"
                ))

            # Data type comparison
            source_dtype = str(source_df[column].dtype)
            target_dtype = str(target_df[column].dtype)

            if source_dtype != target_dtype:
                column_issues.append(ValidationIssue(
                    column=column,
                    issue_type="type_mismatch",
                    message=f"Data type mismatch: {source_dtype} vs {target_dtype}",
                    severity="WARNING"
                ))

            # Unique value count comparison
            source_unique = source_df[column].nunique()
            target_unique = target_df[column].nunique()
            unique_diff = abs(source_unique - target_unique)

            if unique_diff > 0:
                column_issues.append(ValidationIssue(
                    column=column,
                    issue_type="unique_count_mismatch",
                    message=f"Unique count difference: {source_unique} vs {target_unique}",
                    count=unique_diff,
                    severity="INFO"
                ))

            if column_issues:
                issues.extend(column_issues)

            details[column] = {
                'source_null_count': source_nulls,
                'target_null_count': target_nulls,
                'source_null_percentage': source_null_pct,
                'target_null_percentage': target_null_pct,
                'source_dtype': source_dtype,
                'target_dtype': target_dtype,
                'source_unique_count': source_unique,
                'target_unique_count': target_unique
            }

        status = ValidationStatus.FAILED if any(i.severity == "ERROR" for i in issues) else \
                ValidationStatus.WARNING if any(i.severity == "WARNING" for i in issues) else \
                ValidationStatus.PASSED

        message = f"Data integrity validation completed with {len(issues)} issues"

        execution_time = (datetime.now() - start_time).total_seconds()

        return ValidationResultSet(
            validation_type="data_integrity",
            status=status,
            message=message,
            details=details,
            issues=issues,
            execution_time=execution_time,
            timestamp=start_time
        )


class FullDataComparator:
    """Validator for full data comparison"""

    def __init__(self, key_columns: List[str], sample_size: int = 1000):
        """
        Initialize full data comparator

        Args:
            key_columns: List of columns to use as keys for comparison
            sample_size: Number of rows to sample for comparison (for performance)
        """
        self.key_columns = key_columns
        self.sample_size = sample_size

    def validate(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> ValidationResultSet:
        """
        Perform full data comparison between DataFrames

        Args:
            source_df: Source DataFrame (Parquet data)
            target_df: Target DataFrame (SQL data)

        Returns:
            Validation result set
        """
        start_time = datetime.now()
        issues = []

        # Sample data for performance if datasets are large
        if len(source_df) > self.sample_size:
            source_sample = source_df.sample(n=self.sample_size, random_state=42)
        else:
            source_sample = source_df

        if len(target_df) > self.sample_size:
            target_sample = target_df.sample(n=self.sample_size, random_state=42)
        else:
            target_sample = target_df

        # Ensure key columns exist
        missing_keys = set(self.key_columns) - set(source_sample.columns) - set(target_sample.columns)
        if missing_keys:
            return ValidationResultSet(
                validation_type="full_data_comparison",
                status=ValidationStatus.FAILED,
                message=f"Missing key columns: {missing_keys}",
                execution_time=(datetime.now() - start_time).total_seconds()
            )

        try:
            # Merge to find differences
            merged = pd.merge(
                source_sample,
                target_sample,
                on=self.key_columns,
                suffixes=('_source', '_target'),
                how='outer',
                indicator=True
            )

            # Find rows only in source
            source_only = merged[merged['_merge'] == 'left_only']
            if not source_only.empty:
                issues.append(ValidationIssue(
                    column="_merge",
                    issue_type="source_only_rows",
                    message=f"Found {len(source_only)} rows only in source dataset",
                    count=len(source_only),
                    severity="ERROR"
                ))

            # Find rows only in target
            target_only = merged[merged['_merge'] == 'right_only']
            if not target_only.empty:
                issues.append(ValidationIssue(
                    column="_merge",
                    issue_type="target_only_rows",
                    message=f"Found {len(target_only)} rows only in target dataset",
                    count=len(target_only),
                    severity="ERROR"
                ))

            # Find data differences in matching rows
            matching_rows = merged[merged['_merge'] == 'both']
            if not matching_rows.empty:
                for column in set(source_sample.columns) & set(target_sample.columns):
                    if column in self.key_columns:
                        continue

                    source_col = f"{column}_source"
                    target_col = f"{column}_target"

                    if source_col in matching_rows.columns and target_col in matching_rows.columns:
                        # Compare values (handling NaN properly)
                        source_vals = matching_rows[source_col]
                        target_vals = matching_rows[target_col]

                        # Create boolean mask for differences (treating NaN as equal)
                        diff_mask = ~(source_vals.isna() & target_vals.isna()) & \
                                   ((source_vals != target_vals) | (source_vals.isna() != target_vals.isna()))

                        diff_count = diff_mask.sum()
                        if diff_count > 0:
                            issues.append(ValidationIssue(
                                column=column,
                                issue_type="data_mismatch",
                                message=f"Found {diff_count} value differences in {column}",
                                count=diff_count,
                                severity="ERROR",
                                sample_values=matching_rows[diff_mask][source_col].head(5).tolist()
                            ))

        except Exception as e:
            logger.error(f"Error during full data comparison: {e}")
            return ValidationResultSet(
                validation_type="full_data_comparison",
                status=ValidationStatus.ERROR,
                message=f"Error during comparison: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds()
            )

        status = ValidationStatus.FAILED if any(i.severity == "ERROR" for i in issues) else \
                ValidationStatus.WARNING if any(i.severity == "WARNING" for i in issues) else \
                ValidationStatus.PASSED

        message = f"Full data comparison completed with {len(issues)} differences found"

        execution_time = (datetime.now() - start_time).total_seconds()

        return ValidationResultSet(
            validation_type="full_data_comparison",
            status=status,
            message=message,
            details={
                'sample_size_source': len(source_sample),
                'sample_size_target': len(target_sample),
                'key_columns': self.key_columns
            },
            issues=issues,
            execution_time=execution_time,
            timestamp=start_time
        )


class CustomValidationRunner:
    """Runner for custom validation rules"""

    def __init__(self, rules: List[ValidationRule]):
        """
        Initialize custom validation runner

        Args:
            rules: List of custom validation rules
        """
        self.rules = rules

    def validate(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> ValidationResultSet:
        """
        Run custom validation rules

        Args:
            source_df: Source DataFrame (Parquet data)
            target_df: Target DataFrame (SQL data)

        Returns:
            Validation result set
        """
        start_time = datetime.now()
        issues = []
        rule_results = {}

        for rule in self.rules:
            try:
                # Apply rule to both datasets
                source_result = rule.validator(source_df)
                target_result = rule.validator(target_df)

                rule_results[rule.name] = {
                    'source': source_result,
                    'target': target_result,
                    'passed': source_result.get('passed', True) and target_result.get('passed', True)
                }

                if not rule_results[rule.name]['passed']:
                    issues.append(ValidationIssue(
                        column="custom_rule",
                        issue_type=rule.name,
                        message=f"Custom validation rule '{rule.name}' failed: {rule.description}",
                        severity=rule.severity
                    ))

            except Exception as e:
                logger.error(f"Error executing custom rule '{rule.name}': {e}")
                issues.append(ValidationIssue(
                    column="custom_rule",
                    issue_type=rule.name,
                    message=f"Error executing rule '{rule.name}': {str(e)}",
                    severity="ERROR"
                ))

        status = ValidationStatus.FAILED if any(i.severity == "ERROR" for i in issues) else \
                ValidationStatus.WARNING if any(i.severity == "WARNING" for i in issues) else \
                ValidationStatus.PASSED

        message = f"Custom validation rules completed with {len(issues)} failures"

        execution_time = (datetime.now() - start_time).total_seconds()

        return ValidationResultSet(
            validation_type="custom_validation",
            status=status,
            message=message,
            details={'rule_results': rule_results},
            issues=issues,
            execution_time=execution_time,
            timestamp=start_time
        )


class ValidationEngine:
    """Main validation engine that coordinates all validators"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validation engine

        Args:
            config: Validation configuration
        """
        self.config = config
        self.row_count_validator = RowCountValidator(
            tolerance=config.get('row_count_tolerance', 0)
        )
        self.data_integrity_validator = DataIntegrityValidator(
            null_tolerance=config.get('null_value_tolerance', 0.01)
        )
        self.custom_rules = []

    def add_custom_rule(self, rule: ValidationRule):
        """Add a custom validation rule"""
        self.custom_rules.append(rule)

    def validate(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        key_columns: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Perform comprehensive validation between source and target DataFrames

        Args:
            source_df: Source DataFrame (Parquet data)
            target_df: Target DataFrame (SQL data)
            key_columns: Key columns for full comparison (optional)

        Returns:
            Complete validation result
        """
        start_time = datetime.now()
        result = ValidationResult(
            source="parquet_data",
            target="sql_data",
            overall_status=ValidationStatus.PASSED,
            total_rows_source=len(source_df),
            total_rows_target=len(target_df)
        )

        # Row count validation
        if self.config.get('row_count_validation', True):
            row_count_result = self.row_count_validator.validate(source_df, target_df)
            result.add_result(row_count_result)

        # Data integrity validation
        if self.config.get('data_integrity_validation', True):
            integrity_result = self.data_integrity_validator.validate(source_df, target_df)
            result.add_result(integrity_result)

        # Full data comparison (resource intensive)
        if self.config.get('full_data_comparison', False) and key_columns:
            comparator = FullDataComparator(
                key_columns=key_columns,
                sample_size=self.config.get('sample_size', 1000)
            )
            comparison_result = comparator.validate(source_df, target_df)
            result.add_result(comparison_result)

        # Custom validation rules
        if self.config.get('custom_validation_rules', True) and self.custom_rules:
            custom_runner = CustomValidationRunner(self.custom_rules)
            custom_result = custom_runner.validate(source_df, target_df)
            result.add_result(custom_result)

        # Determine overall status
        failed_validations = result.get_failed_validations()
        if failed_validations:
            result.overall_status = ValidationStatus.FAILED
        elif result.get_warnings():
            result.overall_status = ValidationStatus.WARNING

        result.end_time = datetime.now()
        result.metadata['total_execution_time'] = (result.end_time - start_time).total_seconds()

        return result