#! /usr/bin/env python3

import matplotlib.pyplot as plt

def plot_lines(lines):
    # Create a new figure with the size of letter paper (8.5x11 inches)
    fig, ax = plt.subplots(figsize=(8.5, 11))
    
    # Plot each line segment
    for line in lines:
        (start_x, start_y), (end_x, end_y) = line
        ax.plot([start_x, end_x], [start_y, end_y], 'b-')  # Plot line in blue

    # Set limits and labels
    ax.set_xlim(0, 8.5 * 100)  # Assuming 100 pixels per inch
    ax.set_ylim(0, 11 * 100)  # Assuming 100 pixels per inch
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Invert Y-axis to match the image's coordinate system
    plt.gca().invert_yaxis()
    
    # Show the plot
    plt.show()

# Previous testing
import sys
#from finalandDriver.finalpkg_py.scripts.image_proc import ImageProc
from image_proc import *
print(sys.argv)
test = ImageProc(sys.argv[1])
contours = test.get_contours()

lines = test.approximate_splines(contours)


print("Testing image", lines)

plot_lines(lines)


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def detect_edges(image):
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Apply Canny edge detection
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#     return edges

# def find_contours(edges):
#     # Find contours in the edge-detected image
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# def approximate_splines(contours, resolution=0.01):
#     lines = []
#     for contour in contours:
#         # Approximate the contour with a spline
#         epsilon = resolution * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#         for i in range(len(approx) - 1):
#             start = tuple(approx[i][0])
#             end = tuple(approx[i + 1][0])
#             lines.append((start, end))
#     return lines

# def plot_lines(lines):
#     fig, ax = plt.subplots(figsize=(8.5, 11))
#     for line in lines:
#         x_values, y_values = zip(*line)
#         ax.plot(x_values, y_values, 'b-')
#     ax.set_xlim(0, 8.5 * 100)  # Assuming 100 pixels per inch
#     ax.set_ylim(0, 11 * 100)  # Assuming 100 pixels per inch
#     ax.set_xlabel('X-axis')
#     ax.set_ylabel('Y-axis')
#     plt.gca().invert_yaxis()
#     plt.show()

# def main():
#     image_path = 'images/zigzag.jpg'  # Replace with your image path
#     image = cv2.imread(image_path)

#     edges = detect_edges(image)
#     contours = find_contours(edges)
#     lines = approximate_splines(contours)

#     plot_lines(lines)

# if __name__ == "__main__":
#     main()
