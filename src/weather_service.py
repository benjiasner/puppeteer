"""Weather data fetching service using Open-Meteo API."""

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
import urllib.request
import json

from . import config


class WeatherCondition(Enum):
    """Weather condition types."""
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    SNOWY = "snowy"
    PARTLY_CLOUDY = "partly_cloudy"


@dataclass
class DayForecast:
    """Represents a single day's weather forecast."""
    date: str  # Format: "Mon", "Tue", etc.
    temp_high: int
    temp_low: int
    condition: WeatherCondition


@dataclass
class WeatherData:
    """Complete weather data including current conditions and forecast."""
    current_temp: int
    condition: WeatherCondition
    location: str
    forecast: list[DayForecast]  # 7 days


class WeatherService:
    """Service for fetching weather data from Open-Meteo API."""

    # Chicago coordinates
    LATITUDE = 41.8781
    LONGITUDE = -87.6298
    LOCATION_NAME = "Chicago"

    # API URL for Open-Meteo (free, no API key needed)
    API_URL = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&current=temperature_2m,weather_code"
        f"&daily=weather_code,temperature_2m_max,temperature_2m_min"
        f"&temperature_unit=fahrenheit"
        f"&timezone=America/Chicago"
    )

    # WMO Weather interpretation codes
    # https://open-meteo.com/en/docs
    WMO_TO_CONDITION = {
        0: WeatherCondition.SUNNY,      # Clear sky
        1: WeatherCondition.SUNNY,      # Mainly clear
        2: WeatherCondition.PARTLY_CLOUDY,  # Partly cloudy
        3: WeatherCondition.CLOUDY,     # Overcast
        45: WeatherCondition.CLOUDY,    # Foggy
        48: WeatherCondition.CLOUDY,    # Depositing rime fog
        51: WeatherCondition.RAINY,     # Light drizzle
        53: WeatherCondition.RAINY,     # Moderate drizzle
        55: WeatherCondition.RAINY,     # Dense drizzle
        56: WeatherCondition.RAINY,     # Light freezing drizzle
        57: WeatherCondition.RAINY,     # Dense freezing drizzle
        61: WeatherCondition.RAINY,     # Slight rain
        63: WeatherCondition.RAINY,     # Moderate rain
        65: WeatherCondition.RAINY,     # Heavy rain
        66: WeatherCondition.RAINY,     # Light freezing rain
        67: WeatherCondition.RAINY,     # Heavy freezing rain
        71: WeatherCondition.SNOWY,     # Slight snow
        73: WeatherCondition.SNOWY,     # Moderate snow
        75: WeatherCondition.SNOWY,     # Heavy snow
        77: WeatherCondition.SNOWY,     # Snow grains
        80: WeatherCondition.RAINY,     # Slight rain showers
        81: WeatherCondition.RAINY,     # Moderate rain showers
        82: WeatherCondition.RAINY,     # Violent rain showers
        85: WeatherCondition.SNOWY,     # Slight snow showers
        86: WeatherCondition.SNOWY,     # Heavy snow showers
        95: WeatherCondition.RAINY,     # Thunderstorm
        96: WeatherCondition.RAINY,     # Thunderstorm with slight hail
        99: WeatherCondition.RAINY,     # Thunderstorm with heavy hail
    }

    def __init__(self):
        self._cached_data: Optional[WeatherData] = None
        self._cache_time: float = 0
        self._lock = threading.Lock()
        self._fetch_thread: Optional[threading.Thread] = None
        self._running = True

        # Start background fetch
        self._start_background_fetch()

    def _start_background_fetch(self):
        """Start background thread for fetching weather data."""
        self._fetch_thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self._fetch_thread.start()

    def _fetch_loop(self):
        """Background loop that fetches weather data periodically."""
        while self._running:
            self._fetch_weather()
            # Sleep in small increments to allow quick shutdown
            for _ in range(int(config.WEATHER_CACHE_DURATION)):
                if not self._running:
                    break
                time.sleep(1)

    def _wmo_to_condition(self, code: int) -> WeatherCondition:
        """Convert WMO weather code to WeatherCondition."""
        return self.WMO_TO_CONDITION.get(code, WeatherCondition.CLOUDY)

    def _fetch_weather(self):
        """Fetch weather data from Open-Meteo API."""
        try:
            with urllib.request.urlopen(self.API_URL, timeout=10) as response:
                data = json.loads(response.read().decode())

            # Parse current weather
            current_temp = int(round(data["current"]["temperature_2m"]))
            current_code = data["current"]["weather_code"]
            current_condition = self._wmo_to_condition(current_code)

            # Parse 7-day forecast
            forecast = []
            daily = data["daily"]
            for i in range(min(7, len(daily["time"]))):
                date_str = daily["time"][i]
                # Parse date to get day name
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                day_name = date_obj.strftime("%a")  # Mon, Tue, etc.

                forecast.append(DayForecast(
                    date=day_name,
                    temp_high=int(round(daily["temperature_2m_max"][i])),
                    temp_low=int(round(daily["temperature_2m_min"][i])),
                    condition=self._wmo_to_condition(daily["weather_code"][i])
                ))

            weather_data = WeatherData(
                current_temp=current_temp,
                condition=current_condition,
                location=self.LOCATION_NAME,
                forecast=forecast
            )

            with self._lock:
                self._cached_data = weather_data
                self._cache_time = time.time()

        except Exception as e:
            # Silently fail - will use cached data or None
            pass

    def get_weather(self) -> Optional[WeatherData]:
        """Get current weather data (from cache if available)."""
        with self._lock:
            return self._cached_data

    def stop(self):
        """Stop the background fetch thread."""
        self._running = False
        if self._fetch_thread:
            self._fetch_thread.join(timeout=2)
