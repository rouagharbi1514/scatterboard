"""
Local server connector for hotel data ingestion.
Supports direct PostgreSQL or CSV file ingestion.

Environment variables:
    SOURCE_TYPE: Must be set to 'local_server' to use this connector
    LOCAL_DB_HOST, LOCAL_DB_PORT, LOCAL_DB_NAME, LOCAL_DB_USER, LOCAL_DB_PASSWORD:
        PostgreSQL connection details
    LOCAL_CSV_PATH: Path to CSV files for import
    DATA_SOURCE: Either 'postgres' or 'csv'
"""
import os
import csv
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add these imports for Excel support
try:
    import pandas as pd
except ImportError:
    pd = None  # Will raise a more helpful error when actually used

from .storage import dump_raw_to_storage

# Check if we're using PostgreSQL
USE_POSTGRES = os.environ.get("DATA_SOURCE", "").lower() == "postgres"

if USE_POSTGRES:
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        raise ImportError(
            "psycopg2 is required for PostgreSQL connector. "
            "Install it with: pip install psycopg2-binary"
        )

    # Environment variable validation for PostgreSQL
    required_pg_vars = [
        "LOCAL_DB_HOST", "LOCAL_DB_PORT", "LOCAL_DB_NAME",
        "LOCAL_DB_USER", "LOCAL_DB_PASSWORD"
    ]
    for var in required_pg_vars:
        if not os.environ.get(var):
            raise EnvironmentError(f"Required environment variable {var} is not set")
else:
    # Environment variable validation for CSV
    if not os.environ.get("LOCAL_CSV_PATH"):
        raise EnvironmentError("Required environment variable LOCAL_CSV_PATH is not set")

# CSV path
CSV_PATH = os.environ.get("LOCAL_CSV_PATH", "")


def _get_pg_connection():
    """
    Get a PostgreSQL connection.

    Returns:
        psycopg2 connection object

    Raises:
        RuntimeError: If connection fails
    """
    try:
        conn = psycopg2.connect(
            host=os.environ["LOCAL_DB_HOST"],
            port=os.environ["LOCAL_DB_PORT"],
            database=os.environ["LOCAL_DB_NAME"],
            user=os.environ["LOCAL_DB_USER"],
            password=os.environ["LOCAL_DB_PASSWORD"]
        )
        return conn
    except Exception as e:
        raise RuntimeError(f"Failed to connect to PostgreSQL: {e}")


def extract_availability_sql(
    from_date: Optional[datetime] = None,
    days: int = 30
) -> List[Dict[str, Any]]:
    """
    Extract availability data from PostgreSQL.

    Args:
        from_date: Start date for availability data (defaults to today)
        days: Number of days to pull

    Returns:
        List of availability records
    """
    if not USE_POSTGRES:
        raise RuntimeError("PostgreSQL is not configured. Set DATA_SOURCE=postgres.")

    if from_date is None:
        from_date = datetime.utcnow()

    to_date = from_date + timedelta(days=days)

    conn = _get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Adjust SQL query based on your database schema
            cursor.execute("""
                SELECT
                    r.room_id,
                    r.room_type_id,
                    rt.name as room_type_name,
                    a.date,
                    a.available_rooms,
                    a.occupied_rooms,
                    a.out_of_order_rooms,
                    a.rate_amount
                FROM
                    availability a
                JOIN
                    rooms r ON a.room_id = r.room_id
                JOIN
                    room_types rt ON r.room_type_id = rt.room_type_id
                WHERE
                    a.date BETWEEN %s AND %s
                ORDER BY
                    a.date, r.room_type_id
            """, (from_date.date(), to_date.date()))

            # Convert to list of dictionaries
            results = list(cursor)

            # Convert datetime objects to strings for JSON serialization
            for row in results:
                for key, value in row.items():
                    if isinstance(value, (datetime, datetime.date)):
                        row[key] = value.isoformat()

            # Store raw results
            if results:
                dump_raw_to_storage(results, "availability")

            return results
    finally:
        conn.close()


def extract_financial_txns_sql(
    from_dt: Optional[datetime] = None,
    to_dt: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Extract financial transaction data from PostgreSQL.

    Args:
        from_dt: Start datetime for transactions (defaults to 24hr ago)
        to_dt: End datetime for transactions (defaults to now)

    Returns:
        List of financial transaction records
    """
    if not USE_POSTGRES:
        raise RuntimeError("PostgreSQL is not configured. Set DATA_SOURCE=postgres.")

    if from_dt is None:
        from_dt = datetime.utcnow() - timedelta(days=1)
    if to_dt is None:
        to_dt = datetime.utcnow()

    conn = _get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Adjust SQL query based on your database schema
            cursor.execute("""
                SELECT
                    t.transaction_id,
                    t.reservation_id,
                    t.transaction_date,
                    t.amount,
                    t.currency,
                    t.transaction_type,
                    t.payment_method,
                    t.description,
                    r.room_id,
                    r.guest_id,
                    g.name as guest_name
                FROM
                    financial_transactions t
                LEFT JOIN
                    reservations r ON t.reservation_id = r.reservation_id
                LEFT JOIN
                    guests g ON r.guest_id = g.guest_id
                WHERE
                    t.transaction_date BETWEEN %s AND %s
                ORDER BY
                    t.transaction_date DESC
            """, (from_dt, to_dt))

            # Convert to list of dictionaries
            results = list(cursor)

            # Convert datetime objects to strings for JSON serialization
            for row in results:
                for key, value in row.items():
                    if isinstance(value, (datetime, datetime.date)):
                        row[key] = value.isoformat()

            # Store raw results
            if results:
                dump_raw_to_storage(results, "financial_txns")

            return results
    finally:
        conn.close()


def read_data_files(file_pattern: str) -> List[Dict[str, Any]]:
    """
    Read data from CSV or Excel files matching the given pattern.

    Args:
        file_pattern: Glob pattern to match files
                      (e.g., "availability_*.csv", "financial_*.*")

    Returns:
        List of records from the files
    """
    if USE_POSTGRES:
        raise RuntimeError("File import is not configured. Set DATA_SOURCE=csv.")

    # Check for pandas
    if pd is None:
        raise ImportError(
            "pandas is required for Excel file support. "
            "Install it with: pip install pandas openpyxl xlrd"
        )

    full_pattern = os.path.join(CSV_PATH, file_pattern)
    all_results = []

    for file_path in sorted(glob.glob(full_pattern)):
        try:
            file_ext = os.path.splitext(file_path)[1].lower()

            # Process based on file extension
            if file_ext in ['.xlsx', '.xls']:
                # Process Excel files
                df = pd.read_excel(file_path)
                file_records = df.to_dict('records')
                all_results.extend(file_records)
                print(f"Processed Excel file {file_path}: {len(file_records)} records")

            elif file_ext == '.csv':
                # Process CSV files
                with open(file_path, 'r', newline='', encoding='utf-8-sig') as csvfile:
                    # Detect dialect and read CSV
                    sample = csvfile.read(4096)
                    csvfile.seek(0)

                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample)
                    has_header = sniffer.has_header(sample)

                    reader = csv.reader(csvfile, dialect)

                    # Get headers from first row if available
                    headers = next(reader) if has_header else None
                    if not headers:
                        # Generate column names (col_0, col_1, etc.)
                        first_row = next(reader)
                        headers = [f"col_{i}" for i in range(len(first_row))]
                        # Reset file pointer to include this row in the data
                        csvfile.seek(0)
                        next(reader)  # Skip header again

                    # Process rows
                    file_records = []
                    for row in reader:
                        if row:  # Skip empty rows
                            record = {headers[i]: val for i, val in enumerate(row) if i < len(headers)}
                            file_records.append(record)

                    all_results.extend(file_records)
                print(f"Processed CSV file {file_path}: {len(file_records)} records")

            else:
                print(f"Skipping unsupported file type: {file_path}")
                continue

            # Determine data type from file name
            if "availability" in file_path.lower():
                data_type = "availability"
            elif "financial" in file_path.lower() or "transaction" in file_path.lower():
                data_type = "financial_txns"
            else:
                data_type = "unknown"

            # Store raw results from this file
            if file_records:
                dump_raw_to_storage(file_records, data_type)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return all_results


# Alias the old function name to the new one for backward compatibility
read_csv = read_data_files
