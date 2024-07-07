# startup roboter
from NED2.Robot import Robot
from NED2.Plotter import Plotter
from NED2.ImageProcessor import ImageProcessor
from NED2.MazeSolver import MazeSolver
from NED2.MazeScaler import MazeScaler
from NED2.RobotScaler import RobotScaler


def main():
    robot = Robot(simulation=True)
    robot.connect()

    # bring into position
    robot.move_to_observation_pose()

    # take image
    image = robot.take_image()

    plotter = Plotter()
    plotter.imshow(image)

    # solve maze
    processor = ImageProcessor(plotter)
    scaler = MazeScaler()

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

    # display solution
    plotter.maze_solving_overview(maze, maze_solver.get_path(), scaled_maze, path_scaled, left_opening, right_opening)
    plotter.scaled_solution(scaled_maze, path_scaled)

    # run path
    robot_scaler = RobotScaler()
    waypoints = robot_scaler.get_unique_tuples(path_scaled)
    mapped_waypoints = [robot_scaler.map_to_new_coordinate_system(waypoint) for waypoint in waypoints]

    for waypoint in mapped_waypoints:
        robot.move_xy(waypoint[0], waypoint[1])
    # disconnect robot
    robot.disconnect()


if __name__ == '__main__':
    main()
