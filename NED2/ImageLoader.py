import cv2


class ImageLoader:
    def __init__(self, image_path):
        self.image_path = image_path

    def load_image(self):
        return cv2.imread(self.image_path)