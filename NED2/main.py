import numpy as np

from NED2.ImageLoader import ImageLoader
from NED2.ImageProcessor import ImageProcessor
import matplotlib.pyplot as plt

image_path = 'img/real/maze1.png'
loader = ImageLoader(image_path)
image = loader.load_image()

processor = ImageProcessor()
cropped_image = processor.get_field_of_interest(image, padding=-10)

scaled_img = processor.dilation_erosion(cropped_image)
graph = processor.image_to_graph(scaled_img)
maze = processor.remove_white_noise(graph)

from skimage.morphology import skeletonize

def simplify_maze(maze):
    # Convert the maze to boolean values
    maze = maze.astype(bool)

    # Perform skeletonization
    skeleton = skeletonize(maze)

    return skeleton

simplified_maze = simplify_maze(maze)

def remove_zero_rows_and_columns(arr):
    # Remove zero rows
    arr = arr[~np.all(arr == 0, axis=1)]

    # Remove zero columns
    arr = arr[:, ~np.all(arr == 0, axis=0)]

    return arr

maze = remove_zero_rows_and_columns(simplified_maze)

def thicken_lines(image, thickness=3):
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

maze = thicken_lines(maze, 5)

plt.imshow(maze, cmap='gray')
plt.show()