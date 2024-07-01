import unittest
import cv2

from NED2.ImageProcessor import ImageProcessor
from NED2.exception.CircleDetectionError import CircleDetectionError


class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = ImageProcessor()
        self.test_image = cv2.imread('img/test-maze.png')  # replace with your test image path

    def test_get_field_of_interest(self):
        cropped_image = self.processor.get_field_of_interest(self.test_image)
        self.assertIsNotNone(cropped_image)

    def test_get_field_of_interest_no_corners(self):
        no_corners_image = cv2.imread('img/test-no_corner_maze.png')  # replace with your test image path
        with self.assertRaises(CircleDetectionError):
            self.processor.get_field_of_interest(no_corners_image)


if __name__ == '__main__':
    unittest.main()
