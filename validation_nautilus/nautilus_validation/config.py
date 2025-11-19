"""
Configuration management for Nautilus Validation
"""

import os
import yaml
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and environment variable substitution"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager

        Args:
            config_path: Path to configuration file. If None, uses default paths.
        """
        self.config_path = config_path or self._find_config_file()
        self.config = {}
        self._load_config()

    def _find_config_file(self) -> str:
        """Find configuration file in standard locations"""
        possible_paths = [
            os.environ.get('CONFIG_FILE_PATH'),
            'config.yaml',
            'config.yml',
            os.path.expanduser('~/.nautilus/config.yaml'),
            '/etc/nautilus/config.yaml'
        ]

        for path in possible_paths:
            if path and os.path.exists(path):
                return path

        # Default to config.yaml in current directory
        return 'config.yaml'

    def _load_config(self):
        """Load and parse configuration file"""
        try:
            # Load environment variables from .env file if it exists
            load_dotenv()

            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.warning(f"Configuration file {self.config_path} not found, using defaults")
                self.config = self._get_default_config()

            # Substitute environment variables
            self.config = self._substitute_env_vars(self.config)

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = self._get_default_config()

    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in configuration"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            default_value = None

            # Handle default values: ${VAR:default}
            if ':' in env_var:
                env_var, default_value = env_var.split(':', 1)

            return os.getenv(env_var, default_value)
        else:
            return config

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            'digital_ocean': {
                'endpoint': None,
                'region': 'nyc3',
                'spaces_key': '${SPACES_ACCESS_KEY}',
                'spaces_secret': '${SPACES_SECRET_KEY}',
                'bucket_name': 'your-bucket-name'
            },
            'database': {
                'host': 'localhost',
                'port': 3306,
                'username': '${DB_USERNAME}',
                'password': '${DB_PASSWORD}',
                'database_name': 'your_database',
                'charset': 'utf8mb4'
            },
            'validation': {
                'row_count_validation': True,
                'data_integrity_validation': True,
                'full_data_comparison': False,
                'custom_validation_rules': True,
                'batch_size': 10000,
                'max_workers': 4,
                'timeout': 300,
                'row_count_tolerance': 0,
                'null_value_tolerance': 0.01
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'validation.log',
                'max_file_size': '10MB',
                'backup_count': 5
            },
            'reporting': {
                'output_directory': 'reports',
                'format': ['json', 'html'],
                'include_sample_data': True,
                'sample_size': 10
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key: Configuration key (e.g., 'database.host')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_digital_ocean_config(self) -> Dict[str, Any]:
        """Get DigitalOcean Spaces configuration"""
        return self.get('digital_ocean', {})

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.get('database', {})

    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration"""
        return self.get('validation', {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get('logging', {})

    def get_reporting_config(self) -> Dict[str, Any]:
        """Get reporting configuration"""
        return self.get('reporting', {})

    def validate_config(self) -> bool:
        """
        Validate configuration completeness and correctness

        Returns:
            True if configuration is valid, False otherwise
        """
        errors = []

        # Validate DigitalOcean config
        do_config = self.get_digital_ocean_config()
        required_do_keys = ['spaces_key', 'spaces_secret', 'bucket_name']
        for key in required_do_keys:
            if not do_config.get(key) or do_config[key].startswith('${'):
                errors.append(f"Missing required DigitalOcean config: {key}")

        # Validate database config
        db_config = self.get_database_config()
        required_db_keys = ['host', 'username', 'password', 'database_name']
        for key in required_db_keys:
            if not db_config.get(key) or db_config[key].startswith('${'):
                errors.append(f"Missing required database config: {key}")

        if errors:
            for error in errors:
                logger.error(error)
            return False

        return True