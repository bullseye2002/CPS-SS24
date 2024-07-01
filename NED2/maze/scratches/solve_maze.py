import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the uploaded image
image_path = '../img/real/maze1.png'

# Step 1: Load and preprocess the image
def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to binary
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Define a smaller kernel for the morphological operations
    kernel = np.ones((1,1), np.uint8)

    # Perform dilation and erosion with fewer iterations
    dilation = cv2.dilate(binary, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    erosion = erosion[np.any(erosion, axis=1)]

    idx = np.argwhere(np.all(erosion[..., :] == 0, axis=0))
    erosion = np.delete(erosion, idx, axis=1)

    # Apply the masks to the erosion array

    return erosion

# Step 2: Convert the image to a graph representation
def image_to_graph(binary_image):
    h, w = binary_image.shape
    graph = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            if binary_image[y, x] == 255:
                graph[y, x] = 1  # Mark the path

    return graph

# Step 3: Maze-solving algorithm (BFS)
from collections import deque

def bfs_solve_maze(graph):
    h, w = graph.shape
    start = (1, 1)  # Adjust start point to avoid walls
    end = (h-2, w-2)  # Adjust end point to avoid walls

    # Check if start or end points are walls
    if graph[start] == 0 or graph[end] == 0:
        return None

    queue = deque([start])
    came_from = {start: None}

    while queue:
        current = queue.popleft()

        if current == end:
            break

        for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w and graph[neighbor] == 1 and neighbor not in came_from:
                queue.append(neighbor)
                came_from[neighbor] = current

    if end not in came_from:
        return None  # No path found

    # Reconstruct the path
    path = []
    current = end
    while current:
        path.append(current)
        current = came_from[current]

    path.reverse()
    return path

# Step 4: Overlay the solution path on the original image
def draw_solution_path(image_path, path):
    img = cv2.imread(image_path)

    for position in path:
        cv2.circle(img, (position[1], position[0]), 1, (0, 0, 255), -1)

    return img

# Main function to solve the maze
def solve_maze(image_path):
    binary_image = preprocess_image(image_path)

    # vinsualize the binary image
    plt.imshow(binary_image, cmap='gray')
    plt.show()
    graph = image_to_graph(binary_image)

    print(graph)

    path = bfs_solve_maze(graph)

    if path is None:
        print("No path found.")
        return

    solution_image = draw_solution_path(image_path, path)

    # Display the solution image
    plt.imshow(cv2.cvtColor(solution_image, cv2.COLOR_BGR2RGB))
    plt.show()

# Solve the maze with the uploaded image
solve_maze(image_path)