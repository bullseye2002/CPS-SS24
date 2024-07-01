from NED2.ImageLoader import ImageLoader
from NED2.ImageProcessor import ImageProcessor
import matplotlib.pyplot as plt
import cv2

image_path = 'maze/img/real/maze1.png'
loader = ImageLoader(image_path)
image = loader.load_image()

processor = ImageProcessor(image)
cropped_image = processor.get_field_of_interest(padding=-15)

# Display the cropped image
if cropped_image is not None:
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.show()