import time

import numpy as np

from NED2.ImageLoader import ImageLoader
from NED2.ImageProcessor import ImageProcessor
import matplotlib.pyplot as plt

image_path = 'img/real/maze1.png'
loader = ImageLoader(image_path)
image = loader.load_image()

processor = ImageProcessor()
cropped_image = processor.get_field_of_interest(image, padding=-10)

scaled_img = processor.dilation_erosion(cropped_image)
graph = processor.image_to_graph(scaled_img)
black_and_white = processor.remove_white_noise(graph)
maze = processor.simplify_maze(black_and_white)


def thicken_lines(image, thickness=3):
    # Define the structuring element
    # In this case, we're using a 3x3 square which is the simplest structuring element
    kernel = np.ones((thickness, thickness), np.uint8)

    import cv2

    # Check if the image array is empty or not
    if image.size == 0:
        print("The input image is empty.")
        return image

    # Convert the image to uint8 type
    image = image.astype(np.uint8)

    # Use cv2.dilate to thicken the lines in the image
    thick_image = cv2.dilate(image, kernel, iterations=1)

    return thick_image


maze = thicken_lines(maze, 5)

def distance_to_first_one_per_row_left(arr):
    distances = []
    for row in arr:
        indices = np.where(row == 1)
        if indices[0].size > 0:
            distance = indices[0][0]
        else:
            distance = -1  # Return -1 if no 1 is found in the row
        distances.append(distance)
    return distances


def distance_to_first_one_per_row_right(arr):
    distances = []
    for row in arr:
        reversed_row = np.flip(row)  # Reverse the row
        indices = np.where(reversed_row == 1)
        if indices[0].size > 0:
            distance = indices[0][0]
        else:
            distance = -1  # Return -1 if no 1 is found in the row
        distances.append(distance)

    average_distance = np.mean(distances)
    while distances and distances[0] > average_distance:
        distances[0] = 0

    while distances and distances[-1] > average_distance:
        distances[-1] = 0


    average_distance = np.mean(distances)
    # Initialize variables
    current_length = 0
    max_value = 0
    current_value = 0
    max_length = 0
    max_start_index = 0

    # Iterate over the distances
    for i, distance in enumerate(distances):
        if distance > average_distance:
            # If the distance is above the average, increment the current sequence length
            current_length += 1
            current_value += distance
        else:
            # If the distance is below the average or it's the last value, check the current sequence length
            if current_value > max_value:
                max_value = current_value
                max_length = current_length
                max_start_index = i - max_length
            current_length = 0
            current_value = 0

    # Check the last sequence
    if current_length > max_length:
        max_length = current_length
        max_start_index = len(distances) - max_length

    # Plot the data
    fig, ax = plt.subplots()
    ax.plot(distances)

    # Plot the average distance
    ax.axhline(y=average_distance, color='r', linestyle='--')

    # Mark the longest sequence of values above the average
    ax.axvspan(max_start_index, max_start_index + max_length, color='yellow', alpha=0.5)

    # Set the title and labels
    ax.set_title('Distance to First One Per Row with Average')
    ax.set_xlabel('Row')
    ax.set_ylabel('Distance')

    # Display the plot
    plt.show()

    return distances


import matplotlib.pyplot as plt


def get_opening(distances):
    global average_distance, max_length, max_start_index
    import numpy as np
    # Calculate the average distance
    average_distance = np.mean(distances)

    average_distance = np.mean(distances)
    # Initialize variables
    current_length = 0
    max_value = 0
    current_value = 0
    max_length = 0
    max_start_index = 0

    # Iterate over the distances
    for i, distance in enumerate(distances):
        if distance > average_distance:
            # If the distance is above the average, increment the current sequence length
            current_length += 1
            current_value += distance
        else:
            # If the distance is below the average or it's the last value, check the current sequence length
            if current_value > max_value:
                max_value = current_value
                max_length = current_length
                max_start_index = i - max_length
            current_length = 0
            current_value = 0

    # Check the last sequence
    if current_length > max_length:
        max_length = current_length
        max_start_index = len(distances) - max_length
    return max_start_index, max_start_index + max_length


distances_left = distance_to_first_one_per_row_left(maze)
start_left, end_left = get_opening(distances_left)

distances_right = distance_to_first_one_per_row_right(maze)
start_right, end_right = get_opening(distances_right)


def average(num1, num2):
    return round((num1 + num2) / 2)


print(f"Left opening: {start_left} to {end_left} => {average(start_left, end_left)}")
print(f"Right opening: {start_right} to {end_right} => {average(start_right, end_right)}")

h, w = maze.shape
start = (0, average(start_left, end_left))  # Adjust start point to avoid walls
end = (average(start_right, end_right), w-1)

# plot image
plt.imshow(maze, cmap='gray')
plt.show()

import cv2
maze_rgb_bak = cv2.cvtColor(maze * 255, cv2.COLOR_GRAY2RGB)
# #
# # # Draw the start point (in blue)
# cv2.circle(maze_rgb, (int(start[0]), int(start[1])), radius=5, color=(255, 0, 0), thickness=-1)
# plt.imshow(maze_rgb)
# plt.draw()
# #plt.pause(1)  # pause for 3 seconds
#
# plt.clf()
#
# #
# # # Draw the end point (in red)
# cv2.circle(maze_rgb, (int(end[0]), int(end[1])), radius=5, color=(255, 0, 0), thickness=-1)
# #
# # # Display the maze with the start and end points
# plt.imshow(maze_rgb)
# plt.draw()
# #plt.pause(0)

import random
def get_current_path(maze):
    w, h = maze.shape

    N = 4
    R = 100

    generate_tuples = lambda N, R: [(random.randint(0, R), random.randint(0, R)) for _ in range(N)]
    return generate_tuples(N, R)

import cv2

import matplotlib
#matplotlib.use('TkAgg')
#plt.ion()

# Create a figure and axes
# Create a figure and axes
# Create a figure and axes
#fig, ax = plt.subplots()
#
# # Create an image object with initial data and keep a reference to it
#image_obj = ax.imshow(maze_rgb_bak.copy())
#
# for i in range(10):
#     current_path = get_current_path(maze)
#     maze_rgb = maze_rgb_bak.copy()
#     for position in current_path:
#         cv2.circle(maze_rgb, (position[0], position[1]), 1, (0, 0, 255), -1)
#
#     # Update the image data
#     image_obj.set_data(maze_rgb)
#
#     # Redraw the figure
#     fig.canvas.draw()
#     plt.pause(0.5)


def solve_maze(maze, start, end):
    # Directions for up, down, left, right
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Stack for DFS
    stack = [(start, [start])]

    # Set for visited nodes
    visited = set()

    while stack:
        (x, y), path = stack.pop()
        if (x, y) not in visited:
            if (x, y) == end:
                return path
            visited.add((x, y))
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < len(maze)) and (0 <= ny < len(maze[0])) and maze[nx][ny] != 1:
                    stack.append(((nx, ny), path + [(nx, ny)]))
    return None  # No path found
path = solve_maze(maze, start, end)
print(path)
# Draw the original maze
plt.subplot(1, 2, 1)  # 1 row, 2 columns, index 1
plt.imshow(maze_rgb_bak)
plt.title('Original Maze')

# Draw the solution path on the maze
maze_rgb = maze_rgb_bak.copy()

for position in path:
    cv2.circle(maze_rgb, (position[1], position[0]), 1, (0, 0, 255), 0)

# Display the maze with the solution path
plt.subplot(1, 2, 2)  # 1 row, 2 columns, index 2
plt.imshow(maze_rgb)
plt.title('Solved Maze')

plt.show()