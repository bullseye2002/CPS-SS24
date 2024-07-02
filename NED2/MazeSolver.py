from collections import deque
from matplotlib import pyplot as plt
import cv2


class MazeSolver:

    def __init__(self, maze, start, end):
        self.maze = maze
        self.h, self.w = maze.shape
        self.path = []
        self.start = self.getStart(start)
        self.end = self.getEnd(end)

    def average(self, num1, num2):
        return round((num1 + num2) / 2)

    def getStart(self, start):
        return 0, self.average(start[0], start[1])

    def getEnd(self, end):

        return self.average(end[0], end[1]), self.w - 1

    def solve_maze(self):
        # Directions for up, down, left, right
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # Queue for BFS
        queue = deque([(self.start, [self.start])])

        # Set for visited nodes
        visited = set()

        while queue:
            (x, y), path = queue.popleft()
            if (x, y) not in visited:
                if (x, y) == self.end:
                    self.path = path
                    return path
                visited.add((x, y))
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < len(self.maze)) and (0 <= ny < len(self.maze[0])) and self.maze[nx][ny] != 1:
                        queue.append(((nx, ny), path + [(nx, ny)]))
        return None  # No path found

    def visualize_path(self):
        # plot image
        maze_rgb_bak = cv2.cvtColor(self.maze * 255, cv2.COLOR_GRAY2RGB)
        maze_rgb = maze_rgb_bak.copy()
        maze_rgb = 255 - maze_rgb
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, index 1
        plt.imshow(maze_rgb)
        plt.title('Original Maze')
        for position in self.path:
            cv2.circle(maze_rgb, (position[1], position[0]), 1, (0, 0, 255), 0)

        # display the start and end points from self.start and self.end
        cv2.circle(maze_rgb, (self.start[1], self.start[0]), 1, (0, 255, 0), 0)
        cv2.circle(maze_rgb, (self.end[1], self.end[0]), 1, (0, 255, 0), 0)

        # Display the maze with the solution path
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, index 2
        plt.imshow(maze_rgb)
        plt.title('Solved Maze')
        plt.show()
