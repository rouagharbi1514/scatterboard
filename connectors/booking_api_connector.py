"""Booking API Connector

Handles integration with external booking systems and APIs.
Implements standard booking operations with error handling and rate limiting.
"""

import requests
import json
from datetime import datetime
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from requests.exceptions import RequestException
import logging
from .storage import save_to_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BookingRequest:
    """Data class for booking request parameters"""
    check_in: datetime
    check_out: datetime
    room_type: str
    guest_count: int
    guest_name: str
    email: str
    special_requests: Optional[str] = None

class BookingAPIConnector:
    """Handles communication with external booking APIs"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make HTTP request with error handling and rate limiting"""
        try:
            url = f'{self.base_url}/{endpoint.lstrip("/")}'
            response = self.session.request(method, url, json=data)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f'API request failed: {str(e)}')
            raise

    def check_availability(self, check_in: datetime, check_out: datetime, room_type: Optional[str] = None) -> List[Dict]:
        """Check room availability for given dates"""
        data = {
            'check_in': check_in.isoformat(),
            'check_out': check_out.isoformat(),
            'room_type': room_type
        }
        return self._make_request('GET', '/availability', data)

    def create_booking(self, booking: BookingRequest) -> Dict:
        """Create a new booking"""
        data = {
            'check_in': booking.check_in.isoformat(),
            'check_out': booking.check_out.isoformat(),
            'room_type': booking.room_type,
            'guest_count': booking.guest_count,
            'guest_name': booking.guest_name,
            'email': booking.email,
            'special_requests': booking.special_requests
        }
        response = self._make_request('POST', '/bookings', data)
        save_to_cache('bookings', response)
        return response

    def get_booking(self, booking_id: str) -> Dict:
        """Retrieve booking details"""
        return self._make_request('GET', f'/bookings/{booking_id}')

    def update_booking(self, booking_id: str, updates: Dict) -> Dict:
        """Update an existing booking"""
        return self._make_request('PATCH', f'/bookings/{booking_id}', updates)

    def cancel_booking(self, booking_id: str) -> Dict:
        """Cancel a booking"""
        return self._make_request('DELETE', f'/bookings/{booking_id}')

    def get_room_types(self) -> List[Dict]:
        """Get available room types and their details"""
        return self._make_request('GET', '/room-types')

    def get_rates(self, room_type: Optional[str] = None, date: Optional[datetime] = None) -> List[Dict]:
        """Get room rates for specific room types and dates"""
        params = {
            'room_type': room_type,
            'date': date.isoformat() if date else None
        }
        return self._make_request('GET', '/rates', params)

    def get_booking_stats(self, start_date: datetime, end_date: datetime) -> Dict:
        """Get booking statistics for a date range"""
        params = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        }
        return self._make_request('GET', '/stats', params)