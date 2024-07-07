from NED2.ImageLoader import ImageLoader
from NED2.ImageProcessor import ImageProcessor
from NED2.MazeSolver import MazeSolver
from NED2.Plotter import Plotter
from NED2.Scaler import Scaler

loader = ImageLoader('img/real/maze1.png')
plotter = Plotter()
processor = ImageProcessor(plotter)
scaler = Scaler()

image = loader.load_image()
cropped_image = processor.get_field_of_interest(image, padding=-10)

maze = processor.image_to_maze(cropped_image)
maze = scaler.scaled_down_maze(maze)

left_opening = processor.get_opening(maze)
right_opening = processor.get_opening(maze, inverse=True)

maze_solver = MazeSolver(maze, left_opening, right_opening)
maze_solver.solve_maze()

empty_path_maze = scaler.generate_empty_maze_with_path(maze, maze_solver.get_path())
path_scaled = scaler.revert_scaling(empty_path_maze)
scaled_maze = scaler.revert_scaling(maze)

plotter.maze_solving_overview(maze, maze_solver.get_path(), scaled_maze, path_scaled, left_opening, right_opening)
plotter.scaled_solution(scaled_maze, path_scaled)

print(scaled_maze.shape)
path_coordinates = [(i, j) for i in range(path_scaled.shape[0]) for j in range(path_scaled.shape[1]) if path_scaled[i][j] == 1]
print(path_coordinates)