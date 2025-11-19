"""
Database and storage connectors for Nautilus Validation
"""

import os
import logging
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from typing import Dict, Any, List, Optional, Iterator, Tuple
from urllib.parse import urlparse
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine
import tempfile
import io

logger = logging.getLogger(__name__)


class SpacesConnector:
    """Connector for DigitalOcean Spaces (S3-compatible storage)"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DigitalOcean Spaces connector

        Args:
            config: DigitalOcean configuration dictionary
        """
        self.config = config
        self.endpoint = config.get('endpoint')
        self.region = config.get('region', 'nyc3')
        self.spaces_key = config.get('spaces_key')
        self.spaces_secret = config.get('spaces_secret')
        self.bucket_name = config.get('bucket_name')

        if not all([self.spaces_key, self.spaces_secret, self.bucket_name]):
            raise ValueError("Missing required DigitalOcean Spaces configuration")

        self._initialize_client()

    def _initialize_client(self):
        """Initialize boto3 S3 client for DigitalOcean Spaces"""
        try:
            # Use custom endpoint if provided, otherwise construct from region
            if self.endpoint:
                endpoint_url = self.endpoint
            else:
                endpoint_url = f"https://{self.region}.digitaloceanspaces.com"

            self.client = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=self.spaces_key,
                aws_secret_access_key=self.spaces_secret,
                region_name=self.region
            )

            # Test connection
            self.client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Successfully connected to DigitalOcean Spaces bucket: {self.bucket_name}")

        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Failed to connect to DigitalOcean Spaces: {e}")
            raise

    def list_parquet_files(self, prefix: str = "") -> List[str]:
        """
        List Parquet files in the Spaces bucket

        Args:
            prefix: Prefix to filter files (e.g., 'data/2023/')

        Returns:
            List of Parquet file keys
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            files = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                # More flexible parquet detection - check if .parquet appears anywhere in filename
                if '.parquet' in key.lower():
                    files.append(key)

            logger.info(f"Found {len(files)} Parquet files with prefix '{prefix}'")
            return files

        except ClientError as e:
            logger.error(f"Error listing Parquet files: {e}")
            raise

    def read_parquet_file(self, key: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Read a Parquet file from DigitalOcean Spaces

        Args:
            key: S3 key of the Parquet file
            columns: List of columns to read (optional)

        Returns:
            DataFrame containing the Parquet data
        """
        try:
            logger.info(f"Reading Parquet file: {key}")

            # Download file to memory
            response = self.client.get_object(Bucket=self.bucket_name, Key=key)
            parquet_bytes = response['Body'].read()

            # Read Parquet from bytes
            table = pq.read_table(io.BytesIO(parquet_bytes), columns=columns)
            df = table.to_pandas()

            logger.info(f"Successfully read {len(df)} rows from {key}")
            return df

        except ClientError as e:
            logger.error(f"Error reading Parquet file {key}: {e}")
            raise

    def read_parquet_files_batch(
        self,
        keys: List[str],
        columns: Optional[List[str]] = None,
        batch_size: int = 10000
    ) -> Iterator[pd.DataFrame]:
        """
        Read multiple Parquet files in batches

        Args:
            keys: List of S3 keys for Parquet files
            columns: List of columns to read (optional)
            batch_size: Maximum number of rows per batch

        Yields:
            DataFrames containing batched data
        """
        current_batch = pd.DataFrame()
        current_batch_size = 0

        for key in keys:
            try:
                df = self.read_parquet_file(key, columns)

                if current_batch_size + len(df) > batch_size:
                    # Yield current batch and start new one
                    if not current_batch.empty:
                        yield current_batch
                    current_batch = df
                    current_batch_size = len(df)
                else:
                    # Add to current batch
                    current_batch = pd.concat([current_batch, df], ignore_index=True)
                    current_batch_size += len(df)

            except Exception as e:
                logger.warning(f"Error processing file {key}: {e}")
                continue

        # Yield remaining batch
        if not current_batch.empty:
            yield current_batch

    def get_file_info(self, key: str) -> Dict[str, Any]:
        """
        Get information about a file in Spaces

        Args:
            key: S3 key of the file

        Returns:
            Dictionary containing file information
        """
        try:
            response = self.client.head_object(Bucket=self.bucket_name, Key=key)
            return {
                'key': key,
                'size': response.get('ContentLength', 0),
                'last_modified': response.get('LastModified'),
                'etag': response.get('ETag', '').strip('"'),
                'content_type': response.get('ContentType', '')
            }
        except ClientError as e:
            logger.error(f"Error getting file info for {key}: {e}")
            raise


class DatabaseConnector:
    """Connector for MySQL database"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize database connector

        Args:
            config: Database configuration dictionary
        """
        self.config = config
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 3306)
        self.username = config.get('username')
        self.password = config.get('password')
        self.database_name = config.get('database_name')
        self.charset = config.get('charset', 'utf8mb4')

        if not all([self.host, self.username, self.password, self.database_name]):
            raise ValueError("Missing required database configuration")

        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize SQLAlchemy database engine"""
        try:
            connection_string = (
                f"mysql+pymysql://{self.username}:{self.password}@"
                f"{self.host}:{self.port}/{self.database_name}"
                f"?charset={self.charset}"
            )

            self.engine = create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info(f"Successfully connected to MySQL database: {self.database_name}")

        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to MySQL database: {e}")
            raise

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about a database table

        Args:
            table_name: Name of the table

        Returns:
            Dictionary containing table information
        """
        try:
            inspector = inspect(self.engine)

            # Check if table exists
            if table_name not in inspector.get_table_names():
                raise ValueError(f"Table '{table_name}' does not exist in database")

            # Get column information
            columns = inspector.get_columns(table_name)
            column_info = {
                col['name']: {
                    'type': str(col['type']),
                    'nullable': col['nullable'],
                    'default': col.get('default')
                }
                for col in columns
            }

            # Get row count
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`"))
                row_count = result.scalar()

            return {
                'table_name': table_name,
                'row_count': row_count,
                'columns': column_info,
                'column_names': list(column_info.keys())
            }

        except SQLAlchemyError as e:
            logger.error(f"Error getting table info for {table_name}: {e}")
            raise

    def read_table_data(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        batch_size: int = 10000,
        offset: int = 0
    ) -> Iterator[pd.DataFrame]:
        """
        Read data from a database table in batches

        Args:
            table_name: Name of the table
            columns: List of columns to read (optional)
            where_clause: WHERE clause for filtering (optional)
            batch_size: Number of rows per batch
            offset: Starting offset

        Yields:
            DataFrames containing batched data
        """
        try:
            # Build column list
            column_list = ", ".join([f"`{col}`" for col in columns]) if columns else "*"

            # Build base query
            query = f"SELECT {column_list} FROM `{table_name}`"

            if where_clause:
                query += f" WHERE {where_clause}"

            query += f" ORDER BY (SELECT NULL) LIMIT {batch_size} OFFSET {offset}"

            while True:
                with self.engine.connect() as conn:
                    df = pd.read_sql_query(query, conn)

                if df.empty:
                    break

                yield df
                offset += batch_size
                query = query.replace(f"LIMIT {batch_size} OFFSET {offset - batch_size}",
                                     f"LIMIT {batch_size} OFFSET {offset}")

        except SQLAlchemyError as e:
            logger.error(f"Error reading table data from {table_name}: {e}")
            raise

    def get_row_count(self, table_name: str, where_clause: Optional[str] = None) -> int:
        """
        Get row count for a table

        Args:
            table_name: Name of the table
            where_clause: WHERE clause for filtering (optional)

        Returns:
            Row count
        """
        try:
            query = f"SELECT COUNT(*) FROM `{table_name}`"
            if where_clause:
                query += f" WHERE {where_clause}"

            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                return result.scalar()

        except SQLAlchemyError as e:
            logger.error(f"Error getting row count for {table_name}: {e}")
            raise

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a custom SQL query

        Args:
            query: SQL query to execute

        Returns:
            DataFrame containing query results
        """
        try:
            with self.engine.connect() as conn:
                return pd.read_sql_query(query, conn)

        except SQLAlchemyError as e:
            logger.error(f"Error executing query: {e}")
            raise

    def close(self):
        """Close database connection"""
        if hasattr(self, 'engine'):
            self.engine.dispose()
            logger.info("Database connection closed")