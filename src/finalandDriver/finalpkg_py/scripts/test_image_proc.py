#! /usr/bin/env python3

import matplotlib.pyplot as plt
import time
import numpy as np

# def plot_lines(lines, size):
#     # Create a new figure with the size of letter paper (8.5x11 inches)
#     fig, ax = plt.subplots(figsize=(8.5, 11))
    
#     # Plot each line segment
#     for line in lines:
#         (start_x, start_y), (end_x, end_y) = line
#         ax.plot([start_x, end_x], [start_y, end_y], 'b-')  # Plot line in blue

#     # Set limits and labels
#     ax.set_xlabel('X-axis')
#     ax.set_ylabel('Y-axis')

#     # Invert Y-axis to match the image's coordinate system
#     plt.gca().invert_yaxis()
    
#     # Show the plot
#     plt.show()

def plot_lines(lines, size):
    # Create a new figure with the size of letter paper (8.5x11 inches)
    fig, ax = plt.subplots(figsize=(8.5, 11))
    
    # Plot each line segment
    for set in lines:
        for line in set:
            (start_x, start_y), (end_x, end_y) = line
            ax.plot([start_x, end_x], [start_y, end_y], 'b-')  # Plot line in blue

    # Set limits and labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Invert Y-axis to match the image's coordinate system
    plt.gca().invert_yaxis()
    
    # Show the plot
    plt.show()


def compute_optimal_path(lines):
    """Compute the optimal path with the fewest discontinuities

    Parameters
    ----------
    lines : list
        List of line segments, where each segment is represented as a tuple of start and end points

    Returns
    -------
    optimal_path : list
        List of lists where each element represents a continuous set of line segments
    """
    if not lines:
        return []

    print(f'Computing optimal path for {len(lines)} line segments. This might take a while...')

    start_time = time.time()

    # Initialize the optimal path with the first line segment
    optimal_path = [[lines[0]]]
    lines = lines[1:]

    # Convert lines to numpy array for faster computation
    lines_array = np.array(lines)
    while len(lines_array) > 0:
        last_segment = optimal_path[-1][-1]
        last_point = np.array(last_segment[1])

        # Calculate distances from the last point to the start of each segment
        start_points = lines_array[:, 0]
        distances = np.linalg.norm(start_points - last_point, axis=1)

        # Find the closest segment
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        next_segment = lines_array[min_index]

        # If the closest segment is continuous, add it to the current path
        if min_distance < 0.001:
            optimal_path[-1].append(tuple(map(tuple, next_segment)))
        else:
            # Otherwise, start a new path
            optimal_path.append([tuple(map(tuple, next_segment))])

        # Remove the selected segment from the list
        lines_array = np.delete(lines_array, min_index, axis=0)

    print(f'Optimal path computed in {time.time() - start_time:.2f} seconds')

    return optimal_path

# Previous testing
import sys
#from finalandDriver.finalpkg_py.scripts.image_proc import ImageProc
from image_proc import *
import numpy as np
import time
print(sys.argv)
test = ImageProc(sys.argv[1])
contours = test.get_contours()

lines = test.approximate_splines(contours, 0.001)

lines = compute_optimal_path(lines)

for i, segment in enumerate(lines):
    print(f'Segment {i}')
    for j, line in enumerate(segment):
        print(f'Line {j}: {line}')



print("Number of segments:", len(lines))

size = test.get_image_size()

plot_lines(lines, size)


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
