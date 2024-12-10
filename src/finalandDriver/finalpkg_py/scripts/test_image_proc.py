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

def compute_optimal_path_length(lines):
    """Compute the optimal path with the least distance traveled

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
        end_points = lines_array[:, 1]

        start_distances = np.linalg.norm(start_points - last_point, axis=1)
        end_distances = np.linalg.norm(end_points - last_point, axis=1)

        # Find the closest segment
        min_start_index = np.argmin(start_distances)
        min_end_index = np.argmin(end_distances)
        
        if start_distances[min_start_index] <= end_distances[min_end_index]:
            min_index = min_start_index
            min_distance = start_distances[min_start_index]
            next_segment = lines_array[min_index]
        else:
            min_index = min_end_index
            min_distance = end_distances[min_end_index]
            next_segment = [lines_array[min_index][1], lines_array[min_index][0]]

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


def compute_optimal_path_cull(lines):
    """Compute the optimal path with the least distance traveled and remove repeated lines.

    Parameters
    ----------
    lines : list
        List of line segments, where each segment is represented as a tuple of start and end points.

    Returns
    -------
    optimal_path : list
        List of lists where each element represents a continuous set of line segments.
    """
    if not lines:
        return []

    print(f'Computing optimal path for {len(lines)} line segments. This might take a while...')

    start_time = time.time()

    # Remove repeated lines (including reverse order)
    unique_lines = set()
    for line in lines:
        sorted_line = tuple(sorted(line))
        unique_lines.add(sorted_line)
    unique_lines = list(unique_lines)

    # Initialize the optimal path with the first line segment
    optimal_path = [[unique_lines[0]]]
    unique_lines = unique_lines[1:]

    # Convert lines to numpy array for faster computation
    lines_array = np.array(unique_lines)
    while len(lines_array) > 0:
        last_segment = optimal_path[-1][-1]
        last_point = np.array(last_segment[1])

        # Calculate distances from the last point to the start and end of each segment
        start_points = lines_array[:, 0]
        end_points = lines_array[:, 1]

        start_distances = np.linalg.norm(start_points - last_point, axis=1)
        end_distances = np.linalg.norm(end_points - last_point, axis=1)

        # Find the closest segment
        min_start_index = np.argmin(start_distances)
        min_end_index = np.argmin(end_distances)
        
        if start_distances[min_start_index] <= end_distances[min_end_index]:
            min_index = min_start_index
            min_distance = start_distances[min_start_index]
            next_segment = lines_array[min_index]
        else:
            min_index = min_end_index
            min_distance = end_distances[min_end_index]
            next_segment = [lines_array[min_index][1], lines_array[min_index][0]]

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



def compute_optimal_path_longest(lines):
    """Compute the optimal path with the least distance traveled, drawing the longest segments first.

    Parameters
    ----------
    lines : list
        List of line segments, where each segment is represented as a tuple of start and end points.

    Returns
    -------
    optimal_path : list
        List of lists where each element represents a continuous set of line segments.
    """
    if not lines:
        return []

    # Sort lines by length in descending order
    lines = sorted(lines, key=lambda line: np.linalg.norm(np.array(line[1]) - np.array(line[0])), reverse=True)

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
        end_points = lines_array[:, 1]

        start_distances = np.linalg.norm(start_points - last_point, axis=1)
        end_distances = np.linalg.norm(end_points - last_point, axis=1)

        # Find the closest segment
        min_start_index = np.argmin(start_distances)
        min_end_index = np.argmin(end_distances)
        
        if start_distances[min_start_index] <= end_distances[min_end_index]:
            min_index = min_start_index
            min_distance = start_distances[min_start_index]
            next_segment = lines_array[min_index]
        else:
            min_index = min_end_index
            min_distance = end_distances[min_end_index]
            next_segment = [lines_array[min_index][1], lines_array[min_index][0]]

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

contours = test.get_contours_filter_exp(epsilon_factor=0.001, distance_threshold=10, min_length=50)

lines = test.approximate_splines(contours, 0.001)

lines = compute_optimal_path_longest(lines)

for i, segment in enumerate(lines):
    print(f'Segment {i}')
    for j, line in enumerate(segment):
        print(f'Line {j}: {line}')



print("Number of segments:", len(lines))

size = test.get_image_size()

plot_lines(lines, size)
