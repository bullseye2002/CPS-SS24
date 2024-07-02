from NED2.ImageLoader import ImageLoader
from NED2.ImageProcessor import ImageProcessor

from NED2.MazeSolver import MazeSolver

image_path = 'img/real/maze4.png'
loader = ImageLoader(image_path)
image = loader.load_image()

processor = ImageProcessor()
cropped_image = processor.get_field_of_interest(image, padding=-10)

scaled_img = processor.dilation_erosion(cropped_image)
graph = processor.image_to_graph(scaled_img)
black_and_white = processor.remove_white_noise(graph)
maze = processor.simplify_maze(black_and_white)
maze = processor.thicken_lines(maze, 5)

distances_left = processor.distance_to_first_one_per_row(maze)
start_left, end_left = processor.get_opening(distances_left)

distances_right = processor.distance_to_first_one_per_row(maze, True)
start_right, end_right = processor.get_opening(distances_right)


maze_solver = MazeSolver(maze)
start = maze_solver.getStart(start_left, end_left)
end = maze_solver.getEnd(start_right, end_right)
maze_solver.solve_maze(start, end)
maze_solver.visualize_path()
