"""Image manipulation helpers (grid overlay, cropping, etc.)."""

from __future__ import annotations

import io
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid threading issues on macOS
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Coordinate grid overlay
# ---------------------------------------------------------------------------

def add_coordinate_grid(image_bytes: bytes) -> bytes:
    """Return *new* PNG bytes with a coordinate grid super-imposed over *image_bytes*."""
    
    # Load the image from bytes - EXACTLY like original
    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size
    if width == 0 or height == 0:
        logger.warning("Image has zero dimensions, returning original")
        return image_bytes
    img_array = np.array(img)
    logger.info(f"Adding grid to image of size {width}x{height}")
    
    # Calculate figsize to preserve aspect ratio - EXACTLY like original
    aspect = width / float(height)
    base_size = 10.0
    figsize = (base_size * aspect, base_size)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display the image without deformation in its original orientation - EXACTLY like original
    ax.imshow(img_array)
    
    # Set up coordinate system overlay - EXACTLY like original
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert y-axis to have origin at bottom-left
    
    # Create custom tick locations and labels for 0-30 scale - EXACTLY like original
    # Calculate tick positions based on image dimensions
    x_tick_positions = np.linspace(0, width, 31)  # 31 points for 0-30
    y_tick_positions = np.linspace(0, height, 31)
    x_tick_labels = np.arange(0, 31)
    y_tick_labels = np.arange(30, -1, -1)  # Reverse labels to have 0 at bottom
    
    # Set ticks with custom positions and labels - EXACTLY like original
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add axis labels - EXACTLY like original
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Add title - EXACTLY like original
    ax.set_title('Image with Coordinate System (Origin at 0,0)')
    
    # Save the figure to bytes - EXACTLY like original
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=250, bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Cropping based on 0-grid coordinates
# ---------------------------------------------------------------------------

def crop_with_coordinates(image_bytes: bytes, coordinates: Dict[str, Tuple[int, int]]) -> bytes:
    """Crop an image using coordinates in 0-30 scale - EXACTLY like original."""
    
    # Load the original image - EXACTLY like original
    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size
    
    # Extract all coordinate points - EXACTLY like original
    points = [coordinates["A"], coordinates["B"], coordinates["C"], coordinates["D"]]
    
    # Convert from 0-30 scale to pixel coordinates - EXACTLY like original
    pixel_points = []
    for point in points:
        pixel_x = int(point[0] * width / 30)
        pixel_y = int((30 - point[1]) * height / 30)  # Invert Y since image origin is top-left
        pixel_points.append([pixel_x, pixel_y])
    
    # Find bounding box (min/max x and y coordinates) - EXACTLY like original
    x_coords = [p[0] for p in pixel_points]
    y_coords = [p[1] for p in pixel_points]
    
    min_x = max(0, min(x_coords))
    max_x = min(width, max(x_coords))
    min_y = max(0, min(y_coords))
    max_y = min(height, max(y_coords))
    
    # Crop the image - EXACTLY like original
    cropped_img = img.crop((min_x, min_y, max_x, max_y))
    
    # Save to bytes - EXACTLY like original
    buffer = io.BytesIO()
    cropped_img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return buffer.getvalue()
