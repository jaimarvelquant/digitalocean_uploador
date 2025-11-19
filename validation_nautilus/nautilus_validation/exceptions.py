"""
Custom exceptions for Nautilus Validation
"""


class NautilusValidationError(Exception):
    """Base exception for Nautilus Validation"""
    pass


class ConfigurationError(NautilusValidationError):
    """Raised when there's an error in configuration"""
    pass


class ConnectionError(NautilusValidationError):
    """Raised when there's an error connecting to data sources"""
    pass


class DataValidationError(NautilusValidationError):
    """Raised when data validation fails"""
    pass


class DataReadError(NautilusValidationError):
    """Raised when there's an error reading data"""
    pass


class ReportingError(NautilusValidationError):
    """Raised when there's an error generating reports"""
    pass