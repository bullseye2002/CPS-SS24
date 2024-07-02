import cv2
import numpy as np
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt
from NED2.exception.CircleDetectionError import CircleDetectionError


class ImageProcessor:

    def get_field_of_interest(self, image, blur=False, padding=-10):
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

    def dilation_erosion(self, image):
        # Threshold the image to binary
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY_INV)

        # Define a smaller kernel for the morphological operations
        kernel = np.ones((3, 3), np.uint8)

        # Perform dilation and erosion with fewer iterations
        dilation = cv2.dilate(binary, kernel, iterations=3)
        erosion = cv2.erode(dilation, kernel, iterations=3)

        erosion = erosion[np.any(erosion, axis=1)]

        idx = np.argwhere(np.all(erosion[..., :] == 0, axis=0))
        erosion = np.delete(erosion, idx, axis=1)

        # Apply the masks to the erosion array

        return erosion

    def image_to_graph(self, binary_image):
        h, w = binary_image.shape
        graph = np.zeros((h, w), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                if binary_image[y, x] == 255:
                    graph[y, x] = 1  # Mark the path

        return graph

    def remove_white_noise(self, graph):
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

    def simplify_maze(self, maze):
        maze = maze.astype(bool)

        # Perform skeletonization
        skeleton = skeletonize(maze)

        # Remove zero rows
        arr = skeleton[~np.all(skeleton == 0, axis=1)]

        # Remove zero columns
        arr = arr[:, ~np.all(arr == 0, axis=0)]

        return arr

    def thicken_lines(self, image, thickness=3):
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

    def distance_to_first_one_per_row(self, arr, inverse=False):
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


    def plot_distance(self, average_distance, distances, max_length, max_start_index):
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

    def get_opening(self, distances, plot=False):
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