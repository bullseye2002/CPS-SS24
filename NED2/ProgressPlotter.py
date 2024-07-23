import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np


class MazePlotter:
    def __init__(self, maze):
        """
        Initializes the MazePlotter with a 2D maze array.

        Parameters:
        - maze: 2D list or numpy array representing the maze. 1 represents walls, 0 represents paths.
        """
        self.maze = np.array(maze)
        self.waypoints = []

        # Set up the plot
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.maze, cmap='binary')
        self.line, = self.ax.plot([], [], 'ro-')  # Line object for waypoints

    def plot_maze(self):
        """
        Plots the maze with the current waypoints.
        """
        self.im.set_data(self.maze)
        if self.waypoints:
            x, y = zip(*self.waypoints)
            self.line.set_data(y, x)
        else:
            self.line.set_data([], [])
        self.fig.canvas.draw_idle()
        plt.pause(0.0001)  # Pause to update the plot

    def update_waypoints(self, new_waypoints):
        """
        Updates the waypoints to the new given list of points.

        Parameters:
        - new_waypoints: List of tuples representing the new waypoints.
        """
        self.waypoints = new_waypoints
        self.plot_maze()
