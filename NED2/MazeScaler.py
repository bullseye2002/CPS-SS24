import numpy as np


class MazeScaler:

    def __init__(self):
        self.zero_cols = None
        self.zero_cols_index = None
        self.zero_rows = None
        self.zero_rows_index = None

    @staticmethod
    def generate_empty_maze_with_path(maze, path):
        new_maze = np.zeros_like(maze)

        # Iterate over the path
        for position in path:
            # Set the field at the current position to 1
            new_maze[position[0], position[1]] = 1

        return new_maze

    def revert_scaling(self, maze: np.ndarray) -> np.ndarray:
        for i, index in enumerate(self.zero_cols_index):
            maze = np.insert(maze, index, self.zero_cols[:, i], axis=1)

        for i, index in enumerate(self.zero_rows_index):
            maze = np.insert(maze, index, self.zero_rows[i], axis=0)

        return maze

    def scaled_down_maze(self, maze: np.ndarray) -> np.ndarray:
        self.zero_rows_index = np.where(np.all(maze == 0, axis=1))[0]
        self.zero_rows = maze[self.zero_rows_index]
        downscaled_maze = np.delete(maze, self.zero_rows_index, axis=0)

        # Store the indices of all-zero columns
        self.zero_cols_index = np.where(np.all(downscaled_maze == 0, axis=0))[0]
        self.zero_cols = downscaled_maze[:, self.zero_cols_index]
        downscaled_maze = np.delete(downscaled_maze, self.zero_cols_index, axis=1)

        return downscaled_maze
