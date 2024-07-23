import time

from NED2.ImageLoader import ImageLoader
from NED2.Robot import Robot
from NED2.Plotter import Plotter
from NED2.ImageProcessor import ImageProcessor
from NED2.MazeSolver import MazeSolver
from NED2.MazeScaler import MazeScaler
from NED2.RobotScaler import RobotScaler
import logging


def main():
    logger = logging.getLogger(__name__)
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("robot_logs.log"),
                            logging.StreamHandler()
                        ])

    robot = Robot(simulation=False)
    robot.connect()

    # bring into position
    robot.move_to_home_pose()
    robot.move_to_observation_pose()

    # take image
    #image_loader = ImageLoader("img/real/maze1.png")
    image = robot.take_image()
    #image = image_loader.load_image()

    plotter = Plotter()
    plotter.imshow(image)

    # solve maze
    processor = ImageProcessor(plotter)
    scaler = MazeScaler()

    cropped_image = processor.get_field_of_interest(image, padding=-30)
    time.sleep(1)
    maze = processor.image_to_maze(cropped_image)

    plotter.imshow(maze, "image-to_maze")

    maze = scaler.scaled_down_maze(maze)
    plotter.imshow(maze, "scaled_down")

    left_opening = processor.get_opening(maze)
    right_opening = processor.get_opening(maze, inverse=True)

    plotter.maze_with_openings(maze, left_opening, right_opening)

    maze_solver = MazeSolver(maze, left_opening, right_opening)
    maze_solver.solve_maze()

    plotter.maze_with_points(maze, maze_solver.visited, "Visited")
    plotter.maze_with_openings(maze, left_opening, right_opening)
    plotter.maze_with_points_and_lines(maze, maze_solver.get_path(), "Maze solved with purple?")

    empty_path_maze = scaler.generate_empty_maze_with_path(maze, maze_solver.get_path())
    path_scaled = scaler.revert_scaling(empty_path_maze)
    scaled_maze = scaler.revert_scaling(maze)

    # display solution
    plotter.maze_solving_overview(maze, maze_solver.get_path(), scaled_maze, path_scaled, left_opening, right_opening)
    plotter.scaled_solution(scaled_maze, path_scaled)

    scaled_coordinates = scaler.maze_to_coordinates(path_scaled)
    ordered_coordinates = scaler.order_coordinates_by_path(scaled_coordinates)

    # run path
    robot_scaler = RobotScaler()
    waypoints = robot_scaler.get_unique_tuples(ordered_coordinates)
    mapped_waypoints = [robot_scaler.map_coordinates(scaled_maze.shape, waypoint) for waypoint in waypoints]

    logger.info("waypoints")
    logger.info(waypoints)
    plotter.maze_with_points_and_lines(scaled_maze, waypoints, show_only_image=True)

    for waypoint in mapped_waypoints:
        robot.move_xy(waypoint[0], waypoint[1])
        #time.sleep(1)

    # disconnect robot
    try:
        robot.disconnect()
    except Exception as e:
        logger.info(e)


if __name__ == '__main__':
    main()
