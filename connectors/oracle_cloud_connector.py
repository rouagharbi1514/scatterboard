"""
Oracle OPERA Cloud (OHIP) connector for hotel data ingestion.
Handles authentication, API requests, and data storage.

Environment variables:
    SOURCE_TYPE: Must be set to 'oracle_cloud' to use this connector
    OPERA_BASE_URL: Base URL for OHIP API
    OPERA_CLIENT_ID: OAuth client ID
    OPERA_CLIENT_SECRET: OAuth client secret
    OPERA_APP_KEY: Application key for OHIP API
    OPERA_HOTEL_ID: Hotel ID in OPERA system
"""
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import requests
from .storage import dump_raw_to_storage

# Environment variable validation
required_vars = [
    "OPERA_BASE_URL", "OPERA_CLIENT_ID", "OPERA_CLIENT_SECRET",
    "OPERA_APP_KEY", "OPERA_HOTEL_ID"
]

for var in required_vars:
    if not os.environ.get(var):
        raise EnvironmentError(f"Required environment variable {var} is not set")

# Constants
BASE_URL = os.environ["OPERA_BASE_URL"]
CLIENT_ID = os.environ["OPERA_CLIENT_ID"]
CLIENT_SECRET = os.environ["OPERA_CLIENT_SECRET"]
APP_KEY = os.environ["OPERA_APP_KEY"]
HOTEL_ID = os.environ["OPERA_HOTEL_ID"]

# Cache token
_cached_token = None
_token_expiry = 0


def get_token() -> str:
    """
    Get OAuth token for OHIP API access.
    Uses cached token if valid, otherwise gets a new one.

    Returns:
        str: Bearer token for API authentication

    Raises:
        RuntimeError: If token retrieval fails after retries
    """
    global _cached_token, _token_expiry

    # Return cached token if still valid (with 5 min safety margin)
    current_time = time.time()
    if _cached_token and _token_expiry > current_time + 300:
        return _cached_token

    # Token URL typically on same domain but different path
    token_url = f"{BASE_URL}/oauth/v1/tokens"

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "x-app-key": APP_KEY,
    }

    data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }

    # Implement retry logic
    max_retries = 3
    retry_delay = 1  # starting delay in seconds

    for attempt in range(max_retries):
        try:
            response = requests.post(token_url, headers=headers, data=data, timeout=30)
            response.raise_for_status()

            token_data = response.json()
            _cached_token = token_data["access_token"]
            # Set expiry based on expires_in (typically in seconds)
            _token_expiry = current_time + int(token_data.get("expires_in", 3600))

            return _cached_token

        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = retry_delay * (2 ** attempt)
                print(f"Token retrieval failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to retrieve token after {max_retries} attempts: {e}")


def _make_api_request(
    endpoint: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    next_page_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Make API request to OHIP with retry logic.

    Args:
        endpoint: API endpoint (relative to BASE_URL)
        method: HTTP method (GET, POST, etc.)
        params: Query parameters
        data: Request body for POST/PUT requests
        next_page_token: Token for pagination

    Returns:
        API response as JSON

    Raises:
        RuntimeError: If API request fails after retries
    """
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"

    # Get token and set up headers
    token = get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "x-app-key": APP_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Add pagination token if provided
    if params is None:
        params = {}
    if next_page_token:
        params["nextPageToken"] = next_page_token

    # Add hotel ID to params if not explicitly provided
    if "hotelId" not in params:
        params["hotelId"] = HOTEL_ID

    # Convert data to JSON if provided
    json_data = json.dumps(data) if data else None

    # Implement retry logic
    max_retries = 3
    retry_delay = 1  # starting delay in seconds

    for attempt in range(max_retries):
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                data=json_data,
                timeout=60
            )

            # If unauthorized, token might have expired - get a new one and retry
            if response.status_code == 401:
                # Clear cached token to force refresh
                global _cached_token, _token_expiry
                _cached_token = None
                _token_expiry = 0

                # Get new token and update headers
                token = get_token()
                headers["Authorization"] = f"Bearer {token}"
                continue

            response.raise_for_status()
            return response.json()

        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = retry_delay * (2 ** attempt)
                print(f"API request failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"API request failed after {max_retries} attempts: {e}")


def pull_availability(
    from_date: Optional[datetime] = None,
    days: int = 30
) -> List[Dict[str, Any]]:
    """
    Pull availability data from OHIP API.

    Args:
        from_date: Start date for availability data (defaults to today)
        days: Number of days to pull (defaults to 30)

    Returns:
        List of availability records
    """
    if from_date is None:
        from_date = datetime.utcnow()

    to_date = from_date + timedelta(days=days)

    # Format dates for API
    from_date_str = from_date.strftime("%Y-%m-%d")
    to_date_str = to_date.strftime("%Y-%m-%d")

    params = {
        "hotelId": HOTEL_ID,
        "fromDate": from_date_str,
        "toDate": to_date_str
    }

    all_results = []
    next_page_token = None

    # Handle pagination
    while True:
        response = _make_api_request(
            endpoint="/availability/v1/hotels/availability",
            params=params,
            next_page_token=next_page_token
        )

        # Extract data
        if "availability" in response:
            all_results.extend(response["availability"])

        # Check for next page
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    # Store raw results
    if all_results:
        dump_raw_to_storage(all_results, "availability")

    return all_results


def pull_financial_txns(
    from_dt: Optional[datetime] = None,
    to_dt: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Pull financial transaction data from OHIP API.

    Args:
        from_dt: Start datetime for transactions (defaults to 24hr ago)
        to_dt: End datetime for transactions (defaults to now)

    Returns:
        List of financial transaction records
    """
    if from_dt is None:
        from_dt = datetime.utcnow() - timedelta(days=1)
    if to_dt is None:
        to_dt = datetime.utcnow()

    # Format dates for API with ISO format
    from_dt_str = from_dt.isoformat() + "Z"  # UTC timezone
    to_dt_str = to_dt.isoformat() + "Z"  # UTC timezone

    params = {
        "hotelId": HOTEL_ID,
        "fromDate": from_dt_str,
        "toDate": to_dt_str
    }

    all_results = []
    next_page_token = None

    # Handle pagination
    while True:
        response = _make_api_request(
            endpoint="/financial/v1/hotels/transactions",
            params=params,
            next_page_token=next_page_token
        )

        # Extract data - adjust key name based on actual OHIP API response
        if "transactions" in response:
            all_results.extend(response["transactions"])
        elif "financialTransactions" in response:
            all_results.extend(response["financialTransactions"])

        # Check for next page
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    # Store raw results
    if all_results:
        dump_raw_to_storage(all_results, "financial_txns")

    return all_results
