from NED2.ImageLoader import ImageLoader
from NED2.ImageProcessor import ImageProcessor
from NED2.MazeSolver import MazeSolver

image_path = 'img/real/maze1.png'
loader = ImageLoader(image_path)
image = loader.load_image()

processor = ImageProcessor()
cropped_image = processor.get_field_of_interest(image, padding=-10)

maze = processor.image_to_maze(cropped_image)

left_opening = processor.get_opening(maze)
right_opening = processor.get_opening(maze, inverse=True)

maze_solver = MazeSolver(maze, left_opening, right_opening)
maze_solver.solve_maze()
maze_solver.visualize_path()
