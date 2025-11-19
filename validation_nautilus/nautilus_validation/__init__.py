"""
Nautilus Validation - Parquet to SQL Data Validation Tool

A comprehensive tool for validating Parquet data stored in DigitalOcean Spaces
against data in MySQL databases.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .validators import ValidationEngine, ValidationResult
from .connectors import SpacesConnector, DatabaseConnector
from .config import ConfigManager

__all__ = [
    "ValidationEngine",
    "ValidationResult",
    "SpacesConnector",
    "DatabaseConnector",
    "ConfigManager",
]