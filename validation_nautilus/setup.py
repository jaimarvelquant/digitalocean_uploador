from setuptools import setup, find_packages

setup(
    name="nautilus-validation",
    version="1.0.0",
    description="Parquet to SQL data validation tool",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "boto3>=1.34.0",
        "pyarrow>=14.0.2",
        "pandas>=2.1.4",
        "sqlalchemy>=2.0.23",
        "pymysql>=1.1.0",
        "pyyaml>=6.0.1",
        "click>=8.1.7",
        "colorama>=0.4.6",
        "jinja2>=3.1.2",
        "tqdm>=4.66.1",
        "python-dotenv>=1.0.0",
        "tabulate>=0.9.0",
    ],
    entry_points={
        "console_scripts": [
            "nautilus-validate=nautilus_validation.cli:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)