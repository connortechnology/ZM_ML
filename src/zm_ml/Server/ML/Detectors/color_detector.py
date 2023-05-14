"""
This will take an image and return the dominant color which obviously requires a color image.
Use ML to find a car, truck, etc. crop the bounding box and pass the image to this function to get the dominant color.
Understand that if CCTV IR is on, you will only get black/white/grey spectrum.
"""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors

# TODO: Make it a detector and have it initialized during init

# 50x50 resize: Dominant colors (RGB): ['dimgray', 'lightsteelblue', 'black', 'lightgray', 'gray', 'darkslategray'] -- TOTAL: 4 sec
# original size: Dominant colors (RGB): ['lightsteelblue', 'darkslategray', 'dimgray', 'gray', 'black', 'lightgray'] -- TOTAL: 15.1 sec
def get_dominant_colors(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # image = cv2.resize(image, (50, 50))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Reshape the image to a 2D array of pixels and 3 color values (RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # Apply KMeans clustering to find the dominant color
    kmeans.fit(image)
    # Get the most dominant color (with the highest number of points in the cluster)
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    # Get a list of the most dominant colors and sort them by count
    sorts = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
    print(f"{unique=} -- {counts=} -- {sorts=}")
    dominant_colors = kmeans.cluster_centers_[unique]

    # convert to int
    dominant_colors = [tuple(map(int, color)) for color in dominant_colors]
    return dominant_colors

    # dominant_color = kmeans.cluster_centers_[unique[np.argmax(counts)]]
    # return tuple(map(int, dominant_color))


def closest_color(requested_color: tuple):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[rd + gd + bd] = name
    return min_colors[min(min_colors.keys())]


from typing import List


def get_color_name(requested_colors: List[tuple]):
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


kmeans: KMeans

if __name__ == "__main__":
    _gstart = perf_counter()
    image_path = "/home/baudneo/Pictures/ALPR/nissan rouge.jpeg"
    img_path = Path(image_path)
    if img_path.is_file():
        print(f"File {image_path} exists")
    else:
        print(f"File {image_path} does not exist")
        exit(1)
    _start = perf_counter()
    kmeans = KMeans(n_clusters=6, random_state=0)
    _end = perf_counter()
    print(f"KMeans INIT took {_end - _start} seconds")
    dominant_colors = get_dominant_colors(image_path)
    print(
        f"FIRST ROUND: Dominant colors (RGB): {get_color_name(dominant_colors)} -- PERF:: {perf_counter() - _gstart} seconds"
    )
    # reset kmeans
    dominant_colors = get_dominant_colors(image_path)
    nstart = perf_counter()
    print(
        f"SECOND ROUND: Dominant colors (RGB): {get_color_name(dominant_colors)} -- PERF:: {perf_counter() - nstart} seconds"
    )
    print(f"total time: {perf_counter() - _gstart} seconds")
