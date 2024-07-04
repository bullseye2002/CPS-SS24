import array
from typing import Any, Tuple

import cv2
import numpy as np
from numpy import ndarray, dtype
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt

from NED2.exception.CircleDetectionError import CircleDetectionError


class ImageProcessor:

    def __init__(self):
        self.zero_cols_index = None
        self.zero_rows_index = None
        self.zero_cols = None
        self.zero_rows = None
        self.erosion_removed = None
        self.columns_idx = None
        self.rows_removed = None
        self.rows_idx = None

    @staticmethod
    def get_field_of_interest(image: Any, blur: bool = False, padding: int = -10):
        # Convert to grayscale
        current_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve circle detection
        if blur:
            current_image = cv2.GaussianBlur(current_image, (9, 9), 3)

        # Use HoughCircles to detect circles
        circles = cv2.HoughCircles(current_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                   param1=50, param2=30, minRadius=10, maxRadius=20)

        # Filter and draw detected circles based on position
        circle_image = image.copy()
        detected_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Filter out circles based on expected positions (four corners of the image)
                if (x < 150 and y < 150) or (x > 350 and y < 150) or (x < 150 and y > 250) or (x > 350 and y > 250):
                    detected_circles.append((x, y, r))
                    cv2.circle(circle_image, (x, y), r, (0, 255, 0), 4)
                    cv2.rectangle(circle_image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        if len(detected_circles) != 4:
            raise CircleDetectionError(len(detected_circles))

        x_coords = [x for (x, y, r) in detected_circles]
        y_coords = [y for (x, y, r) in detected_circles]

        # Get bounding box coordinates
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        # Crop the image using the bounding box coordinates and add padding
        cropped_image = image[y_min - padding:y_max + padding, x_min - padding:x_max + padding]
        return cropped_image

    def dilation_erosion(self, image: Any) -> ndarray:
        # Threshold the image to binary
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY_INV)

        # Define a smaller kernel for the morphological operations
        kernel = np.ones((3, 3), np.uint8)

        # Perform dilation and erosion with fewer iterations
        dilation = cv2.dilate(binary, kernel, iterations=3)
        erosion = cv2.erode(dilation, kernel, iterations=3)

        rows_kept = self.remove_zero_rows(erosion)
        columns_kept = self.remove_zero_columns(rows_kept)

        return columns_kept

    def remove_zero_rows(self, erosion) -> ndarray:
        self.rows_idx = np.argwhere(np.all(erosion == 0, axis=1))
        self.rows_removed = erosion[self.rows_idx, ...]
        return np.delete(erosion, self.rows_idx, axis=0)

    def remove_zero_columns(self, erosion) -> ndarray:
        self.columns_idx = np.argwhere(np.all(erosion[..., :] == 0, axis=0))
        # remove the all-zero columns
        self.erosion_removed = erosion[..., self.columns_idx]
        return np.delete(erosion, self.columns_idx, axis=1)

    @staticmethod
    def image_to_graph(binary_image: np.ndarray) -> np.ndarray:
        h, w = binary_image.shape
        graph = np.zeros((h, w), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                if binary_image[y, x] == 255:
                    graph[y, x] = 1  # Mark the path

        return graph

    @staticmethod
    def remove_white_noise(graph: np.ndarray) -> np.ndarray:
        # Create a copy of the array to avoid modifying the original array
        arr_copy = graph.copy()
        rows, cols = graph.shape
        for i in range(rows):
            for j in range(cols):
                if graph[i][j] == 1:
                    # Check if there is a 0 to the far right, far left, far top, and far bottom
                    if (0 in graph[i][:j] and 0 in graph[i][j + 1:]) and (0 in graph[:i, j] and 0 in graph[i + 1:, j]):
                        arr_copy[i][j] = 1
                    else:
                        arr_copy[i][j] = 0
        return arr_copy

    def simplify_maze(self, maze: np.ndarray) -> np.ndarray:
        maze = maze.astype(bool)

        # Perform skeletonization
        skeleton = skeletonize(maze)

        # Store the indices of all-zero rows
        self.zero_rows_index = np.where(np.all(skeleton == 0, axis=1))[0]
        self.zero_rows = skeleton[self.zero_rows_index]
        arr = np.delete(skeleton, self.zero_rows_index, axis=0)

        # Store the indices of all-zero columns
        self.zero_cols_index = np.where(np.all(arr == 0, axis=0))[0]
        self.zero_cols = arr[:, self.zero_cols_index]
        arr = np.delete(arr, self.zero_cols_index, axis=1)

        return arr

    def revert_simplify(self, arr: np.ndarray) -> np.ndarray:

        for i, index in enumerate(self.zero_cols_index):
            arr = np.insert(arr, index, self.zero_cols[:, i], axis=1)

        for i, index in enumerate(self.zero_rows_index):
            arr = np.insert(arr, index, self.zero_rows[i], axis=0)

        return arr

    @staticmethod
    def thicken_lines(image: np.ndarray, thickness: int = 3) -> np.ndarray:
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

    @staticmethod
    def distance_to_first_one_per_row(arr: array, inverse: bool = False) -> array:
        """
        Calculate the distance to the first one per row
        :param arr:
        :param inverse: This is used for the right side of the maze
        :return:
        """
        distances = []
        for row in arr:
            if inverse:
                row = np.flip(row)  # Reverse the row
            indices = np.where(row == 1)
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

        return distances

    @staticmethod
    def plot_distance(average_distance: int, distances: np.ndarray, max_length: int, max_start_index: int):
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

    def get_opening_from_distance(self, distances: np.ndarray, plot: bool = False):
        # Calculate the average distance
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

        if plot:
            self.plot_distance(average_distance, distances, max_length, max_start_index)

        return max_start_index, max_start_index + max_length

    def get_opening(self, maze: np.ndarray, inverse: bool = False, plot: bool = False) -> tuple[int, int]:
        distances_left = self.distance_to_first_one_per_row(maze, inverse)
        return self.get_opening_from_distance(distances_left, plot)

    def image_to_maze(self, cropped_image):
        scaled_img = self.dilation_erosion(cropped_image)
        graph = self.image_to_graph(scaled_img)
        black_and_white = self.remove_white_noise(graph)
        maze = self.simplify_maze(black_and_white)

        print(scaled_img.shape, graph.shape, black_and_white.shape, maze.shape)

        # visualize the image
        plt.imshow(maze, cmap='gray')
        plt.show()

        return self.thicken_lines(maze, 5)
