import array
from typing import Tuple


class RobotScaler:
    @staticmethod
    def map_to_new_coordinate_system(coord: Tuple, x_range: Tuple = (0.17, 0.33), y_range: Tuple = (-0.07, 0.07)) -> Tuple:
        # Normalize the coordinates
        normalized_coord = (coord[0] / 243, coord[1] / 258)

        # Map to the new coordinate system
        mapped_coord = (normalized_coord[0] * (x_range[1] - x_range[0]) + x_range[0],
                        normalized_coord[1] * (y_range[1] - y_range[0]) + y_range[0])

        return mapped_coord

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
