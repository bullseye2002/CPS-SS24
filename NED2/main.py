import cv2
import numpy as np
from matplotlib import pyplot as plt

from NED2.ImageLoader import ImageLoader
from NED2.ImageProcessor import ImageProcessor
from NED2.MazeSolver import MazeSolver

image_path = 'img/real/maze1.png'
loader = ImageLoader(image_path)
image = loader.load_image()

processor = ImageProcessor()
cropped_image = processor.get_field_of_interest(image, padding=-10)

maze = processor.image_to_maze(cropped_image)

left_opening = processor.get_opening(maze)
right_opening = processor.get_opening(maze, inverse=True, plot=False)

maze_solver = MazeSolver(maze, left_opening, right_opening)
maze_solver.solve_maze()
#maze_solver.visualize_path()

#reverted_maze = processor.revert_simplify(maze_solver.maze)


new_maze = np.zeros_like(maze)

# Iterate over the path
for position in maze_solver.path:
    # Set the field at the current position to 1
    new_maze[position[0], position[1]] = 1

path_scaled = processor.revert_simplify(new_maze)


maze_rgb_bak = cv2.cvtColor(maze_solver.maze * 255, cv2.COLOR_GRAY2RGB)
maze_rgb = maze_rgb_bak.copy()
maze_rgb = 255 - maze_rgb


plt.figure(figsize=(10, 10))  # You can adjust the figure size as needed

plt.subplot(2, 2, 1)  # 2 rows, 2 columns, index 1
plt.imshow(maze_rgb)
plt.title('Original Maze')

for position in maze_solver.path:
    cv2.circle(maze_rgb, (position[1], position[0]), 1, (0, 0, 255), 0)

cv2.circle(maze_rgb, maze_solver.start, 5, (255, 0, 0), -1)
cv2.circle(maze_rgb, (maze_solver.end[1], maze_solver.end[0]), 5, (255, 0, 255), -1)

plt.subplot(2, 2, 2)  # 2 rows, 2 columns, index 2
plt.imshow(maze_rgb)
plt.title('Solved Maze')

plt.subplot(2, 2, 3)  # 2 rows, 2 columns, index 3
scaled_maze = processor.revert_simplify(maze_solver.maze)
plt.imshow(scaled_maze)
plt.title('Scaled Maze')

plt.subplot(2, 2, 4)  # 2 rows, 2 columns, index 4
plt.imshow(path_scaled)
plt.title('Scaled Maze')

plt.tight_layout()  # This will ensure that the subplots do not overlap
plt.show()


plt.figure(figsize=(20, 20))  # Set the figure size to a large value

plt.imshow(scaled_maze, cmap='gray')
plt.imshow(path_scaled, cmap='hot', alpha=0.9)  # Overlay path_scaled on top of scaled_maze
plt.title('Scaled Maze with Path Overlay')

plt.tight_layout()  # This will ensure that the subplots do not overlap
plt.show()