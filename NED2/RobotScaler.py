import array
from typing import Tuple


class RobotScaler:
    @staticmethod
    def map_to_new_coordinate_system(maze: array, coord: Tuple, x_range: Tuple = (0.31, 0.17),
                                     y_range: Tuple = (0.07, -0.08)) -> Tuple:
        # Normalize the coordinates
        normalized_coord = (coord[0] / maze.shape[0], coord[1] / maze.shape[1])

        # Map to the new coordinate system
        mapped_coord = (normalized_coord[0] * (x_range[1] - x_range[0]) + x_range[0],
                        normalized_coord[1] * (y_range[1] - y_range[0]) + y_range[0])

        # Clamp the coordinates to ensure they are within the specified ranges
        clamped_x = max(min(mapped_coord[0], max(x_range)), min(x_range))
        clamped_y = max(min(mapped_coord[1], max(y_range)), min(y_range))

        return clamped_x, clamped_y

    def map_coordinates(self, maze_shape, coord, x_range=(0.311, 0.153), y_range=(0.0735, -0.082)) -> Tuple[float, float]:
        """
        Maps coordinates from one coordinate system to another.

        Parameters:
        - maze_shape: Tuple[int, int] representing the shape (dimensions) of the maze.
        - coord: Tuple[int, int] representing the coordinates in the original system.
        - x_range: Tuple[float, float] representing the min and max values in the target system for x.
        - y_range: Tuple[float, float] representing the min and max values in the target system for y.

        Returns:
        - Tuple[float, float]: The mapped coordinates in the target coordinate system.
        """
        # Calculate the proportion of the coord within the maze dimensions
        x_prop = coord[0] / (maze_shape[0] - 1)
        y_prop = coord[1] / (maze_shape[1] - 1)

        # Map the proportions to the new coordinate system
        mapped_x = x_range[0] + x_prop * (x_range[1] - x_range[0])
        mapped_y = y_range[0] + y_prop * (y_range[1] - y_range[0])

        return (mapped_x, mapped_y)

    @staticmethod
    def get_unique_tuples(data: array) -> array:
        # Initialize the result list with the first tuple
        result = [data[0]]

        # Iterate over the rest of the tuples
        for i in range(1, len(data)):
            # If both values in the current tuple are different from the previous one
            if data[i][0] != result[-1][0] and data[i][1] != result[-1][1]:
                # Add the current tuple to the result list
                result.append(data[i])

        return result
