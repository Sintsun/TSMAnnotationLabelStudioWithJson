import cv2
import numpy as np
import logging

def expand_crop_bbox(bbox, image_width, image_height, expand_factor=1.0):
    """Expand or shrink bounding box according to expand_factor, keeping center unchanged"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Calculate center point
    center_x = x1 + width / 2
    center_y = y1 + height / 2

    # Adjust width and height according to expand_factor
    new_width = width * expand_factor
    new_height = height * expand_factor

    # Calculate new x1, y1, x2, y2
    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2

    # Ensure bounding box doesn't exceed image boundaries
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(image_width, new_x2)
    new_y2 = min(image_height, new_y2)

    # Log adjusted bounding box
    logging.debug(f"expand_crop_bbox: Input bbox={bbox}, Expand_factor={expand_factor}, "
                  f"Center=({center_x:.2f}, {center_y:.2f}), Output bbox=[{new_x1:.2f}, {new_y1:.2f}, {new_x2:.2f}, {new_y2:.2f}]")

    # Validate bounding box validity
    if new_x2 <= new_x1 or new_y2 <= new_y1:
        logging.warning(f"Invalid bbox after expand_crop_bbox: [{new_x1}, {new_y1}, {new_x2}, {new_y2}]")
        return bbox  # Return original bounding box to avoid errors

    return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

def resize_bbox(bbox, resize_factor, image_width, image_height):
    """
    Resizes a bounding box by a given factor around its center, checks the width to height ratio,
    crops the upper part if the ratio is smaller than 1:2, adjusts the bounding box to be square,
    and clamps it within image boundaries.
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2
    center_y = y1 + height / 2

    # Check if width-to-height ratio is less than 1:2
    if width * 2 < height:
        # Calculate height to crop
        crop_height = height / 2
        # Crop upper part
        y1 += crop_height
        height = height - crop_height
        y2 = y1 + height
        center_y = y1 + height / 2

    # Adjust width and height according to resize_factor
    new_width = width * resize_factor
    new_height = height * resize_factor

    # Make bounding box square, using the larger side as new side length
    max_side = max(new_width, new_height)
    new_width = new_height = max_side

    # Calculate new coordinates
    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2

    # Clamp coordinates within image boundaries
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(image_width, new_x2)
    new_y2 = min(image_height, new_y2)

    # Recalculate width and height to ensure square
    clamped_width = new_x2 - new_x1
    clamped_height = new_y2 - new_y1
    max_side_clamped = min(clamped_width, clamped_height)

    # Adjust to square based on new center point
    new_x1 = center_x - max_side_clamped / 2
    new_y1 = center_y - max_side_clamped / 2
    new_x2 = center_x + max_side_clamped / 2
    new_y2 = center_y + max_side_clamped / 2

    # Final coordinate clamping to prevent overflow from readjustment
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(image_width, new_x2)
    new_y2 = min(image_height, new_y2)

    return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

def crop_upper_bbox(bbox, image_width, image_height, ratio_threshold=2.0):
    """
    Crops the upper part of a bounding box if its height is greater than or equal to
    a specified multiple of its width, and clamps it within image boundaries.
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Check if the height is greater than or equal to the threshold * width
    if height >= ratio_threshold * width:
        # Crop the upper part by reducing the height
        new_height = width * ratio_threshold  # Upper part height is ratio_threshold times the width
        y2 = y1 + new_height  # Adjust the bottom boundary to crop the upper part

    # Ensure the bounding box is still within image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, x2)
    y2 = min(image_height, y2)

    return [int(x1), int(y1), int(x2), int(y2)]

def make_square_bbox(bbox, image_width, image_height):
    """Adjust bounding box to square, using longer side as side length, centered on center point"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    # Choose longer side as square side length
    side_length = max(width, height)

    # Calculate center point
    center_x = x1 + width / 2
    center_y = y1 + height / 2

    # Calculate new x1, y1, x2, y2
    new_x1 = center_x - side_length / 2
    new_y1 = center_y - side_length / 2
    new_x2 = center_x + side_length / 2
    new_y2 = center_y + side_length / 2

    # Ensure bounding box doesn't exceed image boundaries
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(image_width, new_x2)
    new_y2 = min(image_height, new_y2)

    return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

def adjust_to_even_dimensions(width, height):
    # Make width and height even by flooring to nearest even number
    width = width if width % 2 == 0 else width - 1
    height = height if height % 2 == 0 else height - 1
    return max(width, 2), max(height, 2)  # Ensure minimum size to avoid zero
