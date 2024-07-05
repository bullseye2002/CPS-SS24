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
    def maze(maze):
        # visualize the image
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
