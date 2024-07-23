from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


class Plotter:

    def __init__(self):
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        super().__init__()

    @staticmethod
    def scaled_solution(scaled_maze: ndarray, path_scaled: ndarray) -> None:
        plt.figure(figsize=(20, 20))  # Set the figure size to a large value

        plt.imshow(scaled_maze, cmap='gray')
        plt.imshow(path_scaled, cmap='hot', alpha=0.9)  # Overlay path_scaled on top of scaled_maze
        plt.title('Scaled Maze with Path Overlay')

        plt.tight_layout()  # This will ensure that the subplots do not overlap
        plt.show()

    def maze_with_openings(self, maze, start, end):
        maze_rgb = cv2.cvtColor(maze * 255, cv2.COLOR_GRAY2RGB)

        if start is not None:
            cv2.circle(maze_rgb, (start[1], start[0]), 5, self.red, -1)

        if end is not None:
            cv2.circle(maze_rgb, (end[1], end[0]), 5, self.red, -1)

        plt.imshow(maze_rgb)
        plt.title('Maze with start and end')
        plt.show()

        pass

    def maze_solving_overview(self, maze: ndarray, path, scaled_maze, scaled_path, start: Tuple = None,
                              end: Tuple = None) -> None:
        maze_rgb = cv2.cvtColor(maze * 255, cv2.COLOR_GRAY2RGB)

        plt.figure(figsize=(10, 10))  # You can adjust the figure size as needed

        plt.subplot(2, 2, 1)  # 2 rows, 2 columns, index 1
        plt.imshow(maze_rgb)
        plt.title('Original Maze')

        # draw the path on the maze_rgb image
        for position in path:
            cv2.circle(maze_rgb, (position[1], position[0]), 1, self.blue, 0)

        if start is not None:
            cv2.circle(maze_rgb, start, 5, self.red, -1)

        if end is not None:
            cv2.circle(maze_rgb, (end[1], end[0]), 5, self.red, -1)

        plt.subplot(2, 2, 2)  # 2 rows, 2 columns, index 2
        plt.imshow(maze_rgb)
        plt.title('Solved Maze')

        plt.subplot(2, 2, 3)  # 2 rows, 2 columns, index 3
        plt.imshow(scaled_maze)
        plt.title('Scaled Maze')

        plt.subplot(2, 2, 4)  # 2 rows, 2 columns, index 4
        plt.imshow(scaled_path)
        plt.title('Scaled Maze')

        plt.tight_layout()  # This will ensure that the subplots do not overlap
        plt.show()

    @staticmethod
    def imshow(maze, title="", show_only_image=False):
        fig, ax = plt.subplots()
        if show_only_image:
            plt.axis('off')  # Hide the axis
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove the white space
        else:
            plt.title(title)
            plt.axis('on')  # Show the axis

        plt.imshow(maze, cmap='gray')
        plt.show()

    @staticmethod
    def plot_openings(average_distance: int, distances: np.ndarray, max_length: int, max_start_index: int):
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

    def maze_with_points(self, maze, points, title=""):
        maze_copy = maze.copy()
        maze_rgb = cv2.cvtColor(maze_copy * 255, cv2.COLOR_GRAY2RGB)
        for position in points:
            # Convert the coordinates to integers
            x = int(position[0])
            y = int(position[1])
            cv2.circle(maze_rgb, (y, x), 1, self.red, -1)
        plt.imshow(maze_rgb)
        plt.title(title)
        plt.show()

    def maze_with_points_and_lines(self, maze, points, title="", show_only_image=False):
        maze_copy = maze.copy()
        maze_rgb = cv2.cvtColor(maze_copy * 255, cv2.COLOR_GRAY2RGB)
        last_position = None
        for position in points:
            # Convert the coordinates to integers
            x = int(position[0])
            y = int(position[1])

            # Draw line from the last position to the current position in purple
            if last_position is not None:
                last_x = int(last_position[0])
                last_y = int(last_position[1])
                cv2.line(maze_rgb, (last_y, last_x), (y, x), (128, 0, 128), 2)  # Purple color in BGR

            cv2.circle(maze_rgb, (y, x), 3, self.red, 0)

            last_position = position

        fig, ax = plt.subplots()

        if show_only_image:
            plt.axis('off')  # Hide the axis
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        else:
            plt.title(title)
            plt.axis('on')  # Show the axis

        plt.imshow(maze_rgb, cmap='gray')
        plt.show()


    def draw_maze(self, maze):
        if self.img is None:
            self.img = self.ax.imshow(maze, cmap='gray')
        else:
            self.img.set_data(maze)
        plt.draw()
        plt.pause(0.001)  # Pause to allow the plot to update

    def update_maze_with_path(self, maze, path):
        maze_with_path = np.copy(maze)
        for x, y in path:
            maze_with_path[x, y] = 0.5  # Assuming 0.5 to represent the path visually
        self.img.set_data(maze_with_path)
        plt.draw()
        plt.pause(0.001)  # Pause to allow the plot to update

