"""
Data connectors for hotel dashboard.
Provides interfaces for different data sources.
"""
import os

# Determine which connector to use based on environment variable
SOURCE_TYPE = os.environ.get("SOURCE_TYPE", "").lower()

if SOURCE_TYPE == "oracle_cloud":
    from .oracle_cloud_connector import pull_availability, pull_financial_txns
elif SOURCE_TYPE == "local_server":
    # Import appropriate function based on datasource configuration
    if os.environ.get("DATA_SOURCE", "").lower() == "postgres":
        from .local_server_connector import extract_availability_sql as pull_availability
        from .local_server_connector import extract_financial_txns_sql as pull_financial_txns
    else:
        from .local_server_connector import (
            read_data_files,
        )

        # Define wrapper functions that call read_csv with appropriate patterns
        def pull_availability(from_date=None, days=30):
            return read_data_files("availability_*.*")  # Note the *.* pattern to match all extensions

        def pull_financial_txns(from_dt=None, to_dt=None):
            return read_data_files("financial_*.*")
else:
    # Default implementation for testing/development
    def pull_availability(*args, **kwargs):
        raise NotImplementedError(
            "No data source configured. Set SOURCE_TYPE environment variable."
        )

    def pull_financial_txns(*args, **kwargs):
        raise NotImplementedError(
            "No data source configured. Set SOURCE_TYPE environment variable."
        )


def read_csv(filepath, **kwargs):
    return read_data_files(
        filepath, file_format="csv", **kwargs
    )
