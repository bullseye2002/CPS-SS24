from collections import deque
from matplotlib import pyplot as plt
import cv2


class MazeSolver:

    def __init__(self, maze):
        self.maze = maze
        self.h, self.w = maze.shape
        self.path = []

    def average(self, num1, num2):
        return round((num1 + num2) / 2)

    def getStart(self, start_left, end_left):
        return 0, self.average(start_left, end_left)

    def getEnd(self, start_right, end_right):
        return self.average(start_right, end_right), self.w - 1

    def solve_maze(self, start, end):
        # Directions for up, down, left, right
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # Queue for BFS
        queue = deque([(start, [start])])

        # Set for visited nodes
        visited = set()

        while queue:
            (x, y), path = queue.popleft()
            if (x, y) not in visited:
                if (x, y) == end:
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
        plt.imshow(self.maze, cmap='gray')
        plt.show()
        maze_rgb_bak = cv2.cvtColor(self.maze * 255, cv2.COLOR_GRAY2RGB)
        maze_rgb = maze_rgb_bak.copy()
        maze_rgb = 255 - maze_rgb
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, index 1
        plt.imshow(maze_rgb)
        plt.title('Original Maze')
        for position in self.path:
            cv2.circle(maze_rgb, (position[1], position[0]), 1, (0, 0, 255), 0)
        # Display the maze with the solution path
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, index 2
        plt.imshow(maze_rgb)
        plt.title('Solved Maze')
        plt.show()
