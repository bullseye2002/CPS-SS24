import cv2
import numpy as np

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
