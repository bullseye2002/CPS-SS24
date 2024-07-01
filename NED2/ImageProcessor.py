import cv2
import numpy as np


class ImageProcessor:
    def __init__(self, image):
        self.image = image

    def get_field_of_interest(self, blur=False, padding=-10):
        # Convert to grayscale
        current_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve circle detection
        if blur:
            current_image = cv2.GaussianBlur(current_image, (9, 9), 3)

        # Use HoughCircles to detect circles
        circles = cv2.HoughCircles(current_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                   param1=50, param2=30, minRadius=10, maxRadius=20)

        # Filter and draw detected circles based on position
        circle_image = self.image.copy()
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
            raise ValueError("Could not detect the field of interest in the image.")

        x_coords = [x for (x, y, r) in detected_circles]
        y_coords = [y for (x, y, r) in detected_circles]

        # Get bounding box coordinates
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        # Crop the image using the bounding box coordinates and add padding
        cropped_image = self.image[y_min - padding:y_max + padding, x_min - padding:x_max + padding]
        return cropped_image
