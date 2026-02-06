"""Base widget class for interactive UI elements."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class WidgetState(Enum):
    """Widget interaction states."""
    IDLE = "idle"
    HOVER = "hover"
    EXPANDED = "expanded"


@dataclass
class WidgetBounds:
    """Represents widget boundaries with interaction helpers."""
    x: int
    y: int
    width: int
    height: int

    def contains(self, px: int, py: int) -> bool:
        """Check if a point is inside the bounds."""
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)

    def expanded(self, margin: int) -> "WidgetBounds":
        """Return new bounds expanded by margin on all sides."""
        return WidgetBounds(
            x=self.x - margin,
            y=self.y - margin,
            width=self.width + margin * 2,
            height=self.height + margin * 2
        )


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b by factor t."""
    return a + (b - a) * t


class BaseWidget(ABC):
    """Abstract base class for interactive widgets."""

    # Animation constants
    HOVER_SCALE = 1.08
    IDLE_SCALE = 1.0
    ANIMATION_SPEED = 8.0  # Multiplier for delta_time

    def __init__(self, x: int, y: int, width: int, height: int):
        self._base_x = x
        self._base_y = y
        self._base_width = width
        self._base_height = height

        self._state = WidgetState.IDLE
        self._current_scale = 1.0
        self._target_scale = 1.0
        self._halo_intensity = 0.0
        self._target_halo = 0.0
        self._is_expanded = False
        self._expand_progress = 0.0  # 0 = compact, 1 = fully expanded

    @property
    def bounds(self) -> WidgetBounds:
        """Get current widget bounds (accounting for scale)."""
        # Calculate scaled dimensions
        scaled_width = int(self._base_width * self._current_scale)
        scaled_height = int(self._base_height * self._current_scale)

        # Keep position fixed at top-left
        return WidgetBounds(
            x=self._base_x,
            y=self._base_y,
            width=scaled_width,
            height=scaled_height
        )

    @property
    def is_hovering(self) -> bool:
        """Check if widget is being hovered."""
        return self._state == WidgetState.HOVER

    @property
    def is_expanded(self) -> bool:
        """Check if widget is expanded."""
        return self._is_expanded

    def set_hover(self, hovering: bool):
        """Set hover state."""
        if hovering:
            self._state = WidgetState.HOVER
            self._target_scale = self.HOVER_SCALE
            self._target_halo = 1.0
        else:
            self._state = WidgetState.IDLE
            self._target_scale = self.IDLE_SCALE
            self._target_halo = 0.3  # Dim halo when not hovering

    def toggle_expanded(self):
        """Toggle between compact and expanded states."""
        self._is_expanded = not self._is_expanded

    def update_animation(self, delta_time: float):
        """Update animation state based on elapsed time."""
        # Smooth scale animation
        t = min(1.0, self.ANIMATION_SPEED * delta_time)
        self._current_scale = lerp(self._current_scale, self._target_scale, t)

        # Smooth halo animation
        self._halo_intensity = lerp(self._halo_intensity, self._target_halo, t)

        # Smooth expand animation
        target_expand = 1.0 if self._is_expanded else 0.0
        self._expand_progress = lerp(self._expand_progress, target_expand, t * 0.5)

    @abstractmethod
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the widget on the frame. Returns modified frame."""
        pass

    @abstractmethod
    def on_snap_gesture(self):
        """Handle snap gesture when hovering over widget."""
        pass
