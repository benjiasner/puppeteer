"""OpenCV utilities for frosted glass appearance."""

import cv2
import numpy as np


def create_rounded_rect_mask(
    width: int,
    height: int,
    radius: int
) -> np.ndarray:
    """
    Create a rounded rectangle mask.

    Args:
        width: Mask width
        height: Mask height
        radius: Corner radius

    Returns:
        Single-channel mask with 255 inside, 0 outside
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # Clamp radius to half of smallest dimension
    radius = min(radius, width // 2, height // 2)

    # Draw filled rounded rectangle using multiple shapes
    # Main rectangle (excluding corners)
    cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1)
    cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1)

    # Corner circles
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, height - radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, height - radius), radius, 255, -1)

    return mask


def draw_glass_panel(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    radius: int = 15,
    blur_kernel: int = 25,
    tint_color: tuple = (255, 255, 255),
    tint_opacity: float = 0.2,
    border_opacity: float = 0.4
) -> np.ndarray:
    """
    Draw a frosted glass panel on the frame.

    Args:
        frame: Input BGR frame
        x, y: Top-left corner position
        width, height: Panel dimensions
        radius: Corner radius
        blur_kernel: Gaussian blur kernel size (must be odd)
        tint_color: BGR color for glass tint
        tint_opacity: Opacity of tint overlay (0-1)
        border_opacity: Opacity of highlight border (0-1)

    Returns:
        Frame with glass panel drawn
    """
    frame_h, frame_w = frame.shape[:2]

    # Clamp panel to frame bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_w, x + width)
    y2 = min(frame_h, y + height)

    if x2 <= x1 or y2 <= y1:
        return frame

    # Adjust width/height to actual clipped region
    actual_width = x2 - x1
    actual_height = y2 - y1

    # Extract ROI
    roi = frame[y1:y2, x1:x2].copy()

    # Apply Gaussian blur for frosted effect
    blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred = cv2.GaussianBlur(roi, (blur_kernel, blur_kernel), 0)

    # Create tint overlay
    tint = np.full_like(roi, tint_color, dtype=np.uint8)

    # Blend blurred ROI with tint
    glass = cv2.addWeighted(blurred, 1.0 - tint_opacity, tint, tint_opacity, 0)

    # Create rounded rectangle mask
    mask = create_rounded_rect_mask(actual_width, actual_height, radius)

    # Apply mask to blend glass panel with original frame
    mask_3ch = cv2.merge([mask, mask, mask])
    mask_norm = mask_3ch.astype(np.float32) / 255.0

    # Composite: original * (1 - mask) + glass * mask
    original_roi = frame[y1:y2, x1:x2].astype(np.float32)
    glass_float = glass.astype(np.float32)
    composited = original_roi * (1 - mask_norm) + glass_float * mask_norm

    frame[y1:y2, x1:x2] = composited.astype(np.uint8)

    # Draw highlight borders (top and left edges for depth)
    # Top highlight
    highlight_color = (255, 255, 255)
    pts_top = np.array([
        [x1 + radius, y1 + 1],
        [x2 - radius, y1 + 1]
    ], np.int32)
    overlay = frame.copy()
    cv2.line(overlay, tuple(pts_top[0]), tuple(pts_top[1]), highlight_color, 1, cv2.LINE_AA)

    # Left highlight
    pts_left = np.array([
        [x1 + 1, y1 + radius],
        [x1 + 1, y2 - radius]
    ], np.int32)
    cv2.line(overlay, tuple(pts_left[0]), tuple(pts_left[1]), highlight_color, 1, cv2.LINE_AA)

    # Blend highlight
    cv2.addWeighted(overlay, border_opacity, frame, 1 - border_opacity, 0, frame)

    return frame


def draw_halo_effect(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    radius: int = 15,
    intensity: float = 0.5,
    halo_size: int = 20,
    halo_color: tuple = (255, 200, 100)
) -> np.ndarray:
    """
    Draw a soft glow/halo effect around a widget area.

    Args:
        frame: Input BGR frame
        x, y: Top-left corner position
        width, height: Widget dimensions
        radius: Corner radius
        intensity: Glow intensity (0-1)
        halo_size: Size of glow in pixels
        halo_color: BGR color for the halo

    Returns:
        Frame with halo effect drawn
    """
    if intensity <= 0:
        return frame

    frame_h, frame_w = frame.shape[:2]

    # Create larger mask for halo area
    halo_width = width + halo_size * 2
    halo_height = height + halo_size * 2
    halo_x = x - halo_size
    halo_y = y - halo_size

    # Clamp to frame bounds
    x1 = max(0, halo_x)
    y1 = max(0, halo_y)
    x2 = min(frame_w, halo_x + halo_width)
    y2 = min(frame_h, halo_y + halo_height)

    if x2 <= x1 or y2 <= y1:
        return frame

    actual_width = x2 - x1
    actual_height = y2 - y1

    # Create halo mask using distance transform
    # First create inner shape mask
    inner_mask = np.zeros((actual_height, actual_width), dtype=np.uint8)

    # Offset for the inner rectangle within the halo area
    inner_x = max(0, x - x1)
    inner_y = max(0, y - y1)
    inner_w = min(width, actual_width - inner_x)
    inner_h = min(height, actual_height - inner_y)

    if inner_w > 0 and inner_h > 0:
        # Draw inner rounded rect
        inner_rect_mask = create_rounded_rect_mask(inner_w, inner_h, radius)
        inner_mask[inner_y:inner_y + inner_h, inner_x:inner_x + inner_w] = inner_rect_mask

    # Invert and compute distance transform
    inverted = 255 - inner_mask
    dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)

    # Normalize distance to create falloff (closer = brighter)
    # Only show halo within halo_size distance
    halo_mask = np.clip(1.0 - (dist / halo_size), 0, 1)

    # Don't show halo inside the widget
    halo_mask[inner_mask > 0] = 0

    # Apply intensity
    halo_mask = halo_mask * intensity * 0.5  # Scale down for subtlety

    # Create colored halo
    halo_layer = np.zeros((actual_height, actual_width, 3), dtype=np.float32)
    halo_layer[:, :] = halo_color

    # Extract ROI and blend
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    halo_mask_3ch = np.stack([halo_mask] * 3, axis=-1)

    # Additive blend for glow effect
    result = roi + halo_layer * halo_mask_3ch
    result = np.clip(result, 0, 255)

    frame[y1:y2, x1:x2] = result.astype(np.uint8)

    return frame
