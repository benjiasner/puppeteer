"""Weather widget with frosted glass appearance."""

from typing import Optional

import cv2
import numpy as np

from .. import config
from ..hand_tracker import HandData
from ..weather_service import WeatherCondition, WeatherData, WeatherService
from .base import BaseWidget, WidgetBounds, lerp
from .glass_effects import draw_glass_panel, draw_halo_effect


class WeatherWidget(BaseWidget):
    """Interactive weather widget with glass effect and gesture support."""

    def __init__(self):
        # Initialize at compact size
        compact_w, compact_h = config.WEATHER_WIDGET_COMPACT_SIZE
        super().__init__(
            x=config.WEATHER_WIDGET_MARGIN,
            y=config.WEATHER_WIDGET_MARGIN,
            width=compact_w,
            height=compact_h
        )

        # Weather service
        self._weather_service = WeatherService()

        # Expanded size
        self._expanded_width, self._expanded_height = config.WEATHER_WIDGET_EXPANDED_SIZE

        # Color scheme
        self._text_color = (255, 255, 255)
        self._secondary_color = (200, 200, 200)
        self._halo_color = (255, 200, 100)  # Warm orange glow

    def _get_current_size(self) -> tuple[int, int]:
        """Get interpolated size based on expand progress."""
        compact_w, compact_h = config.WEATHER_WIDGET_COMPACT_SIZE
        expanded_w, expanded_h = config.WEATHER_WIDGET_EXPANDED_SIZE

        width = int(lerp(compact_w, expanded_w, self._expand_progress))
        height = int(lerp(compact_h, expanded_h, self._expand_progress))

        return width, height

    @property
    def bounds(self) -> WidgetBounds:
        """Get current widget bounds (accounting for scale and expand state)."""
        width, height = self._get_current_size()

        # Apply hover scale
        scaled_width = int(width * self._current_scale)
        scaled_height = int(height * self._current_scale)

        return WidgetBounds(
            x=self._base_x,
            y=self._base_y,
            width=scaled_width,
            height=scaled_height
        )

    def check_hover(self, hands_data: list[HandData]) -> bool:
        """
        Check if right index finger is hovering over widget.

        Args:
            hands_data: List of HandData from hand tracker

        Returns:
            True if hovering
        """
        bounds = self.bounds.expanded(10)  # Slightly larger hit area

        for hand in hands_data:
            if hand.handedness == "Right":
                for ft in hand.fingertips:
                    if ft.finger_name == "Index":
                        if bounds.contains(ft.x, ft.y):
                            return True
        return False

    def update(self, hands_data: list[HandData], delta_time: float):
        """
        Update widget state based on hand data and time.

        Args:
            hands_data: List of HandData from hand tracker
            delta_time: Time since last frame in seconds
        """
        # Check hover state
        is_hovering = self.check_hover(hands_data)
        self.set_hover(is_hovering)

        # Update animations
        self.update_animation(delta_time)

    def on_snap_gesture(self):
        """Handle snap gesture - toggle expanded state."""
        self.toggle_expanded()

    def _draw_weather_icon(
        self,
        frame: np.ndarray,
        condition: WeatherCondition,
        cx: int,
        cy: int,
        size: int,
        color: tuple = (255, 255, 255)
    ):
        """
        Draw a weather icon using OpenCV primitives.

        Args:
            frame: Frame to draw on
            condition: Weather condition type
            cx, cy: Center position
            size: Icon size (radius for sun, etc.)
            color: Icon color (BGR)
        """
        if condition == WeatherCondition.SUNNY:
            # Draw sun - circle with rays
            cv2.circle(frame, (cx, cy), size // 2, color, -1, cv2.LINE_AA)
            # Draw rays
            for angle in range(0, 360, 45):
                rad = np.radians(angle)
                x1 = int(cx + (size // 2 + 3) * np.cos(rad))
                y1 = int(cy + (size // 2 + 3) * np.sin(rad))
                x2 = int(cx + (size // 2 + 8) * np.cos(rad))
                y2 = int(cy + (size // 2 + 8) * np.sin(rad))
                cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        elif condition == WeatherCondition.CLOUDY:
            # Draw cloud - overlapping circles
            cv2.circle(frame, (cx - size // 3, cy), size // 3, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx + size // 4, cy), size // 3, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy - size // 4), size // 3, color, -1, cv2.LINE_AA)

        elif condition == WeatherCondition.PARTLY_CLOUDY:
            # Sun behind cloud
            sun_color = (0, 200, 255)  # Yellow-orange
            cv2.circle(frame, (cx + size // 3, cy - size // 4), size // 3, sun_color, -1, cv2.LINE_AA)
            # Cloud in front
            cv2.circle(frame, (cx - size // 4, cy + size // 6), size // 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx + size // 6, cy + size // 6), size // 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx - size // 8, cy - size // 8), size // 4, color, -1, cv2.LINE_AA)

        elif condition == WeatherCondition.RAINY:
            # Cloud with rain drops
            cv2.circle(frame, (cx - size // 4, cy - size // 4), size // 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx + size // 6, cy - size // 4), size // 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx - size // 8, cy - size // 2), size // 5, color, -1, cv2.LINE_AA)
            # Rain drops
            rain_color = (255, 200, 150)  # Light blue
            for i in range(-1, 2):
                x = cx + i * (size // 3)
                cv2.line(frame, (x, cy + size // 6), (x - 3, cy + size // 2), rain_color, 2, cv2.LINE_AA)

        elif condition == WeatherCondition.SNOWY:
            # Cloud with snowflakes
            cv2.circle(frame, (cx - size // 4, cy - size // 4), size // 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx + size // 6, cy - size // 4), size // 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx - size // 8, cy - size // 2), size // 5, color, -1, cv2.LINE_AA)
            # Snowflakes (small circles)
            snow_color = (255, 255, 255)
            for i in range(-1, 2):
                x = cx + i * (size // 3)
                cv2.circle(frame, (x, cy + size // 3), 3, snow_color, -1, cv2.LINE_AA)

    def _draw_compact(self, frame: np.ndarray, weather: Optional[WeatherData]):
        """Draw compact widget view."""
        bounds = self.bounds
        x, y = bounds.x, bounds.y
        width, height = bounds.width, bounds.height

        # Draw halo effect
        draw_halo_effect(
            frame, x, y, width, height,
            radius=15,
            intensity=self._halo_intensity,
            halo_size=15,
            halo_color=self._halo_color
        )

        # Draw glass panel
        draw_glass_panel(
            frame, x, y, width, height,
            radius=15,
            blur_kernel=25,
            tint_opacity=0.25,
            border_opacity=0.3
        )

        if weather is None:
            # Loading state
            cv2.putText(
                frame, "Loading...",
                (x + 15, y + height // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._secondary_color, 1, cv2.LINE_AA
            )
            return

        # Temperature (large)
        temp_text = f"{weather.current_temp}"
        cv2.putText(
            frame, temp_text,
            (x + 15, y + 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, self._text_color, 2, cv2.LINE_AA
        )

        # Degree symbol
        # Get text size to position degree symbol
        (text_w, _), _ = cv2.getTextSize(temp_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
        cv2.putText(
            frame, "F",
            (x + 20 + text_w, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._secondary_color, 1, cv2.LINE_AA
        )

        # Weather icon
        icon_x = x + width - 35
        icon_y = y + 35
        self._draw_weather_icon(frame, weather.condition, icon_x, icon_y, 25, self._text_color)

        # Location
        cv2.putText(
            frame, weather.location,
            (x + 15, y + height - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self._secondary_color, 1, cv2.LINE_AA
        )

    def _draw_expanded(self, frame: np.ndarray, weather: Optional[WeatherData]):
        """Draw expanded widget view with 7-day forecast."""
        bounds = self.bounds
        x, y = bounds.x, bounds.y
        width, height = bounds.width, bounds.height

        # Draw halo effect
        draw_halo_effect(
            frame, x, y, width, height,
            radius=15,
            intensity=self._halo_intensity,
            halo_size=20,
            halo_color=self._halo_color
        )

        # Draw glass panel
        draw_glass_panel(
            frame, x, y, width, height,
            radius=15,
            blur_kernel=25,
            tint_opacity=0.25,
            border_opacity=0.3
        )

        if weather is None:
            cv2.putText(
                frame, "Loading weather...",
                (x + 20, y + height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self._secondary_color, 1, cv2.LINE_AA
            )
            return

        # Current weather header
        # Temperature
        temp_text = f"{weather.current_temp}"
        cv2.putText(
            frame, temp_text,
            (x + 20, y + 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.8, self._text_color, 2, cv2.LINE_AA
        )

        (text_w, _), _ = cv2.getTextSize(temp_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 2)
        cv2.putText(
            frame, "F",
            (x + 25 + text_w, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self._secondary_color, 1, cv2.LINE_AA
        )

        # Location and condition
        cv2.putText(
            frame, weather.location,
            (x + 20, y + 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._secondary_color, 1, cv2.LINE_AA
        )

        # Large weather icon
        icon_x = x + width - 50
        icon_y = y + 45
        self._draw_weather_icon(frame, weather.condition, icon_x, icon_y, 35, self._text_color)

        # Divider line
        cv2.line(
            frame,
            (x + 15, y + 95),
            (x + width - 15, y + 95),
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # 7-day forecast grid
        if weather.forecast:
            forecast_y_start = y + 115
            day_width = (width - 30) // 7

            for i, day in enumerate(weather.forecast[:7]):
                day_x = x + 15 + i * day_width
                day_cx = day_x + day_width // 2

                # Day name
                cv2.putText(
                    frame, day.date,
                    (day_cx - 12, forecast_y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, self._secondary_color, 1, cv2.LINE_AA
                )

                # Weather icon (small)
                self._draw_weather_icon(
                    frame, day.condition,
                    day_cx, forecast_y_start + 35,
                    15, self._text_color
                )

                # High temp
                cv2.putText(
                    frame, f"{day.temp_high}",
                    (day_cx - 10, forecast_y_start + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self._text_color, 1, cv2.LINE_AA
                )

                # Low temp
                cv2.putText(
                    frame, f"{day.temp_low}",
                    (day_cx - 10, forecast_y_start + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, self._secondary_color, 1, cv2.LINE_AA
                )

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the weather widget on the frame."""
        weather = self._weather_service.get_weather()

        # Interpolate between compact and expanded views
        if self._expand_progress < 0.1:
            self._draw_compact(frame, weather)
        elif self._expand_progress > 0.9:
            self._draw_expanded(frame, weather)
        else:
            # During transition, just draw the current size with compact content
            # This creates a smooth size transition
            self._draw_expanded(frame, weather)

        return frame

    def stop(self):
        """Stop the weather service."""
        self._weather_service.stop()
