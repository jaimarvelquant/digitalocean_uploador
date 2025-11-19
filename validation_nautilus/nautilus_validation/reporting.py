"""
Reporting functionality for Nautilus Validation
"""

import os
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
from jinja2 import Template
import pandas as pd
from .validators import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


class ValidationReporter:
    """Generates validation reports in various formats"""

    def __init__(self, output_dir: str = "reports", config: Dict[str, Any] = None):
        """
        Initialize validation reporter

        Args:
            output_dir: Directory to save reports
            config: Reporting configuration
        """
        self.output_dir = output_dir
        self.config = config or {}
        self.include_sample_data = self.config.get('include_sample_data', True)
        self.sample_size = self.config.get('sample_size', 10)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def generate_report(
        self,
        validation_result: ValidationResult,
        formats: List[str] = None,
        filename_prefix: str = None
    ) -> Dict[str, str]:
        """
        Generate validation reports in specified formats

        Args:
            validation_result: Validation result to report
            formats: List of formats to generate (json, html, csv)
            filename_prefix: Prefix for report files

        Returns:
            Dictionary mapping formats to file paths
        """
        formats = formats or self.config.get('format', ['json', 'html'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix = filename_prefix or f"validation_report_{timestamp}"

        report_files = {}

        for format_type in formats:
            try:
                if format_type.lower() == 'json':
                    file_path = self._generate_json_report(validation_result, filename_prefix)
                    report_files['json'] = file_path
                elif format_type.lower() == 'html':
                    file_path = self._generate_html_report(validation_result, filename_prefix)
                    report_files['html'] = file_path
                elif format_type.lower() == 'csv':
                    file_path = self._generate_csv_report(validation_result, filename_prefix)
                    report_files['csv'] = file_path
                else:
                    logger.warning(f"Unsupported report format: {format_type}")

            except Exception as e:
                logger.error(f"Error generating {format_type} report: {e}")

        logger.info(f"Generated reports: {list(report_files.values())}")
        return report_files

    def _generate_json_report(self, validation_result: ValidationResult, filename_prefix: str) -> str:
        """Generate JSON format report"""
        file_path = os.path.join(self.output_dir, f"{filename_prefix}.json")

        # Convert validation result to dictionary
        report_data = validation_result.to_dict()

        # Add metadata
        report_data['report_metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'generator': 'Nautilus Validation',
            'version': '1.0.0'
        }

        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        return file_path

    def _generate_html_report(self, validation_result: ValidationResult, filename_prefix: str) -> str:
        """Generate HTML format report"""
        file_path = os.path.join(self.output_dir, f"{filename_prefix}.html")

        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nautilus Validation Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .status {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 4px;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.8em;
        }
        .status.passed { background-color: #d4edda; color: #155724; }
        .status.failed { background-color: #f8d7da; color: #721c24; }
        .status.warning { background-color: #fff3cd; color: #856404; }
        .status.error { background-color: #f8d7da; color: #721c24; }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .summary-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            text-align: center;
        }
        .summary-card h3 {
            margin: 0 0 10px 0;
            color: #495057;
        }
        .summary-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        .validation-section {
            margin-bottom: 30px;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            overflow: hidden;
        }
        .validation-header {
            background: #f8f9fa;
            padding: 15px;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .validation-body {
            padding: 20px;
        }
        .issues {
            margin-top: 20px;
        }
        .issue {
            background: #fff5f5;
            border-left: 4px solid #e53e3e;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        .issue.warning {
            background: #fffbf0;
            border-left-color: #dd6b20;
        }
        .issue.info {
            background: #ebf8ff;
            border-left-color: #3182ce;
        }
        .issue-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .issue-details {
            font-size: 0.9em;
            color: #666;
        }
        .sample-values {
            margin-top: 10px;
            padding: 10px;
            background: #f7fafc;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.8em;
        }
        .metadata {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 6px;
        }
        .metadata h3 {
            margin-top: 0;
        }
        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        .metadata-item {
            padding: 10px;
            background: white;
            border-radius: 4px;
        }
        .metadata-label {
            font-weight: bold;
            color: #495057;
        }
        .metadata-value {
            color: #6c757d;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }
        th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        .execution-time {
            color: #6c757d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Nautilus Validation Report</h1>
            <p>Generated on {{ generated_at }}</p>
        </div>

        <div class="summary">
            <div class="summary-card">
                <h3>Overall Status</h3>
                <div class="status {{ validation_result.overall_status.value.lower() }}">
                    {{ validation_result.overall_status.value }}
                </div>
            </div>
            <div class="summary-card">
                <h3>Total Rows (Source)</h3>
                <div class="value">{{ validation_result.total_rows_source }}</div>
            </div>
            <div class="summary-card">
                <h3>Total Rows (Target)</h3>
                <div class="value">{{ validation_result.total_rows_target }}</div>
            </div>
            <div class="summary-card">
                <h3>Validations Run</h3>
                <div class="value">{{ validation_result.validation_results|length }}</div>
            </div>
        </div>

        {% for result in validation_result.validation_results %}
        <div class="validation-section">
            <div class="validation-header">
                <div>
                    <h3>{{ result.validation_type.replace('_', ' ').title() }}</h3>
                    <span class="status {{ result.status.value.lower() }}">{{ result.status.value }}</span>
                </div>
                <div class="execution-time">⏱️ {{ "%.3f"|format(result.execution_time) }}s</div>
            </div>
            <div class="validation-body">
                <p><strong>{{ result.message }}</strong></p>

                {% if result.details %}
                <div>
                    <h4>Details:</h4>
                    <table>
                        {% for key, value in result.details.items() %}
                        <tr>
                            <td><strong>{{ key }}</strong></td>
                            <td>{{ value }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                {% endif %}

                {% if result.issues %}
                <div class="issues">
                    <h4>Issues Found:</h4>
                    {% for issue in result.issues %}
                    <div class="issue {{ issue.severity.lower() }}">
                        <div class="issue-title">{{ issue.issue_type }} - {{ issue.column }}</div>
                        <div class="issue-details">
                            {{ issue.message }}
                            {% if issue.count > 0 %} (Count: {{ issue.count }}){% endif %}
                        </div>
                        {% if issue.sample_values %}
                        <div class="sample-values">
                            <strong>Sample values:</strong> {{ issue.sample_values|join(', ') }}
                        </div>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
            </div>
        </div>
        {% endfor %}

        <div class="metadata">
            <h3>📊 Validation Metadata</h3>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <div class="metadata-label">Start Time</div>
                    <div class="metadata-value">{{ validation_result.start_time }}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">End Time</div>
                    <div class="metadata-value">{{ validation_result.end_time }}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Total Execution Time</div>
                    <div class="metadata-value">{{ "%.3f"|format(validation_result.metadata.total_execution_time) }}s</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Source Dataset</div>
                    <div class="metadata-value">{{ validation_result.source }}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Target Dataset</div>
                    <div class="metadata-value">{{ validation_result.target }}</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
        """

        # Prepare template data
        template_data = {
            'validation_result': validation_result,
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Render template
        template = Template(html_template)
        html_content = template.render(**template_data)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return file_path

    def _generate_csv_report(self, validation_result: ValidationResult, filename_prefix: str) -> str:
        """Generate CSV format report"""
        file_path = os.path.join(self.output_dir, f"{filename_prefix}.csv")

        # Prepare CSV data
        csv_data = []

        for result in validation_result.validation_results:
            base_row = {
                'validation_type': result.validation_type,
                'status': result.status.value,
                'message': result.message,
                'execution_time': result.execution_time,
                'timestamp': result.timestamp.isoformat(),
                'issues_count': len(result.issues)
            }

            if result.issues:
                for issue in result.issues:
                    row = base_row.copy()
                    row.update({
                        'issue_column': issue.column,
                        'issue_type': issue.issue_type,
                        'issue_message': issue.message,
                        'issue_count': issue.count,
                        'issue_severity': issue.severity,
                        'sample_values': ';'.join(str(v) for v in issue.sample_values) if issue.sample_values else ''
                    })
                    csv_data.append(row)
            else:
                csv_data.append(base_row)

        # Convert to DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(file_path, index=False)

        return file_path

    def generate_summary_report(self, validation_results: List[ValidationResult]) -> str:
        """
        Generate a summary report for multiple validation results

        Args:
            validation_results: List of validation results

        Returns:
            Path to summary report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.output_dir, f"validation_summary_{timestamp}.json")

        summary_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator': 'Nautilus Validation',
                'version': '1.0.0',
                'total_validations': len(validation_results)
            },
            'summary': {
                'total_validations': len(validation_results),
                'passed': len([r for r in validation_results if r.overall_status == ValidationStatus.PASSED]),
                'failed': len([r for r in validation_results if r.overall_status == ValidationStatus.FAILED]),
                'warnings': len([r for r in validation_results if r.overall_status == ValidationStatus.WARNING]),
                'total_rows_source': sum(r.total_rows_source for r in validation_results),
                'total_rows_target': sum(r.total_rows_target for r in validation_results)
            },
            'validation_results': [result.to_dict() for result in validation_results]
        }

        with open(file_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)

        return file_path