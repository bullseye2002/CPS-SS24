# Load the image
image_path = '../../img/real/maze1.png'
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve circle detection
blurred = cv2.GaussianBlur(gray, (9, 9), 3)

# Use HoughCircles to detect circles
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
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
if len(detected_circles) == 4:
    x_coords = [x for (x, y, r) in detected_circles]
    y_coords = [y for (x, y, r) in detected_circles]

    # Get bounding box coordinates
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # Crop the image using the bounding box coordinates
    padding = -10  # Adjust this value to your needs

    # Crop the image using the bounding box coordinates and add padding
    cropped_image = image[y_min-padding:y_max+padding, x_min-padding:x_max+padding]

    # Display the original image, image with detected circles, and the cropped image
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title('Detected Circles')
    plt.imshow(cv2.cvtColor(circle_image, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 3)
    plt.title('Cropped Image')
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    plt.show()
else:
    print("Exactly four circles were not detected. Please adjust the detection parameters.")