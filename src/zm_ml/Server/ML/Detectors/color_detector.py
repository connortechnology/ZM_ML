"""
This will take an image and return the dominant color which obviously requires a color image.
Use ML to find a car, truck, etc. crop the bounding box and pass the image to this function to get the dominant color.
Understand that if CCTV IR is on, you will only get black/white/grey spectrum.
"""

from pathlib import Path
from time import perf_counter
from typing import List
from logging import getLogger
from warnings import warn
try:
    import cv2
except ImportError:
    warn("OpenCV not installed!")
    raise
import numpy as np
from sklearn.cluster import KMeans
import webcolors

from ..file_locks import FileLock

LP: str = "color detect:"
from ...Log import SERVER_LOGGER_NAME
logger = getLogger(SERVER_LOGGER_NAME)
# TODO: Make it a detector and have it initialized during init

class ColorDetector(FileLock):
    kmeans: KMeans

    def __init__(self, config):
        _start = perf_counter()
        self.kmeans = KMeans(n_clusters=6, random_state=0)
        _end = perf_counter()
        logger.info(f"perf:{LP} init took {_end - _start:.4f} seconds")

    # 50x50 resize: Dominant colors (RGB): ['dimgray', 'lightsteelblue', 'black', 'lightgray', 'gray', 'darkslategray'] -- TOTAL: 4 sec
    # original size: Dominant colors (RGB): ['lightsteelblue', 'darkslategray', 'dimgray', 'gray', 'black', 'lightgray'] -- TOTAL: 15.1 sec
    def detect(self, image: np.ndarray):
        # Load the image
        # image = cv2.imread(image: np.ndarray)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Reshape the image to a 2D array of pixels and 3 color values (RGB)
        image = cv2.resize(image, (50, 50))
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        # Apply KMeans clustering to find the dominant color
        self.kmeans.fit(image)
        # Get the most dominant color (with the highest number of points in the cluster)
        unique, counts = np.unique(self.kmeans.labels_, return_counts=True)
        # Get a list of the most dominant colors and sort them by count
        sorts = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
        logger.debug(f"{LP} {unique=} -- {counts=} -- {sorts=}")
        dominant_colors = self.kmeans.cluster_centers_[unique]
        # convert to int
        dominant_colors = [tuple(map(int, color)) for color in dominant_colors]
        colors = self.get_color_name(dominant_colors)
        logger.debug(
            f"{LP} Dominant colors (RGB): {colors}"
        )


        return self.get_color_name(dominant_colors)
    def closest_color(self, requested_color: tuple):
        min_colors = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_color[0]) ** 2
            gd = (g_c - requested_color[1]) ** 2
            bd = (b_c - requested_color[2]) ** 2
            min_colors[rd + gd + bd] = name
        return min_colors[min(min_colors.keys())]

    def get_color_name(self, requested_colors: List[tuple]):
        try:
            color_names = [
                webcolors.rgb_to_name(requested_color)
                for requested_color in requested_colors
            ]
            # color_name = webcolors.rgb_to_name(requested_colors)
        except ValueError:
            color_names = [
                closest_color(requested_color) for requested_color in requested_colors
            ]
        return color_names
