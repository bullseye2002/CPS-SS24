class CircleDetectionError(Exception):
    def __init__(self, num_circles):
        self.num_circles = num_circles
        self.message = f"Circle detection error: Detected {self.num_circles} circles, expected 4."
        super().__init__(self.message)