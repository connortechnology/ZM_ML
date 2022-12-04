import logging
from typing import List, Dict, Tuple, Optional

from shapely.geometry import Polygon

import cv2
import numpy as np

logger = logging.getLogger("ZM_ML-Client")

SLATE_COLORS: List[Tuple[int, int, int]] = [
    (39, 174, 96),
    (142, 68, 173),
    (0, 129, 254),
    (254, 60, 113),
    (243, 134, 48),
    (91, 177, 47),
]


def draw_filtered_bboxes(
    image: np.ndarray,
    filtered_bboxes: List[List[int]],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draws the bounding boxes on the image.

    Args:
        image (np.ndarray): The image to draw on.
        filtered_bboxes (List[Dict]): The filtered bounding boxes.
        color (Tuple[int, int, int]): The color to use [Default: Red].
        thickness (int, optional): The thickness of the bounding boxes. Defaults to 2.

    Returns:
        np.ndarray: The image with the bounding boxes drawn.
    """
    # image = image.copy()
    lp = f"image::draw filtered bbox::"
    for bbox in filtered_bboxes:
        logger.debug(f"{lp} drawing {bbox}")
        # draw bounding box around object
        cv2.rectangle(
            image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness
        )
    return image


def draw_zones(
    image: np.ndarray,
    zone_polygons: List[Polygon],
    poly_color: Tuple[int, int, int] = (255, 255, 255),
    poly_line_thickness: int = 1,
):
    lp: str = "image::zones::"
    # image = image.copy()
    if poly_line_thickness:
        for zone_polygon in zone_polygons:
            try:
                """
                Syntax: cv2.polylines(image, [pts], isClosed, color, thickness) 

                image: It is the image on which circle is to be drawn. 
                
                pts: Array of polygonal curves. 
                
                npts: Array of polygon vertex counters. 
                
                ncontours: Number of curves. 
                
                isClosed: Flag indicating whether the drawn polylines are closed or not. If they are closed, the function draws a line from the last vertex of each curve to its first vertex. 
                
                color: It is the color of polyline to be drawn. For BGR, we pass a tuple. 
                
                thickness: It is thickness of the polyline edges. 
                """
                cv2.polylines(
                    image,
                    [np.asarray(list(zip(*zone_polygon.exterior.coords.xy))[:-1])],
                    True,
                    poly_color,
                    poly_line_thickness,
                )
            except Exception as exc:
                logger.error(f"{lp} could not draw polygon -> {exc}")
                return
            else:
                logger.debug(f"{lp} Successfully drew polygon")
    return image


def draw_bounding_boxes(
    image: np.ndarray,
    labels: List[str],
    confidences: List[float],
    boxes: List[List[int]],
    model: str,
    processor: str,
    write_conf: bool = False,
    write_model: bool = False,
    write_processor: bool = False,
) -> np.ndarray:
    """Draw bounding boxes and labels on an image"""
    # FIXME: need to add scaling dependant on image dimensions
    bgr_slate_colors = SLATE_COLORS[::-1]
    w, h = image.shape[:2]
    lp = f"image::draw bbox::"
    arr_len = len(bgr_slate_colors)
    i = 0
    for label, conf, bbox in zip(labels, confidences, boxes):
        logger.debug(f"{lp} drawing '{label}' bounding box @ {bbox}")
        bbox_color = bgr_slate_colors[i % arr_len]
        i += 1
        if write_conf and conf:
            label = f"{label} {round(conf * 100)}%"
        if write_model and model:
            if write_processor and processor:
                label = f"{label} [{model}::{processor}]"

            else:
                label = f"{label} [{model}]"
        # draw bounding box around object
        cv2.rectangle(
            image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, 2
        )

        # write label
        font_thickness = 1
        font_scale = 0.6
        # FIXME: add something better than this
        if w >= 720:
            # 720p+
            font_scale = 1.0
            font_thickness = 2
        elif w >= 1080:
            # 1080p+
            font_scale = 1.7
            font_thickness = 2
        elif w >= 1880:
            # 3-4k ish? +
            font_scale = 3.2
            font_thickness = 4

        logger.debug(
            f"{lp} ({i}/{len(labels)}) w*h={(w, h)} {font_scale=} {font_thickness=} {bbox=} "
        )
        font_type = cv2.FONT_HERSHEY_DUPLEX
        text_size = cv2.getTextSize(label, font_type, font_scale, font_thickness)[0]
        text_width_padded = text_size[0] + 4
        text_height_padded = text_size[1] + 4
        r_top_left = (bbox[0], bbox[1] - text_height_padded)
        r_bottom_right = (bbox[0] + text_width_padded, bbox[1])
        # Background
        cv2.rectangle(image, r_top_left, r_bottom_right, bbox_color, -1)
        # cv2.putText(image, text, (x, y), font, font_scale, color, thickness)

        # Make sure the text is inside the image
        if r_bottom_right[0] > w:
            r_bottom_right = (w, r_bottom_right[1])
            r_top_left = (r_bottom_right[0] - text_width_padded, r_top_left[1])
        if r_bottom_right[1] > h:
            r_bottom_right = (r_bottom_right[0], h)
            r_top_left = (r_top_left[0], r_bottom_right[1] - text_height_padded)

        cv2.putText(
            image,
            label,
            (bbox[0] + 2, bbox[1] - 2),
            font_type,
            font_scale,
            [255, 255, 255],
            font_thickness,
            cv2.LINE_AA,
        )

    return image
