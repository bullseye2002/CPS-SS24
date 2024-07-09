from collections import deque
import logging


class MazeSolver:

    def __init__(self, maze, start, end):
        self.visited = None
        self.maze = maze
        self.h, self.w = maze.shape
        self.path = []
        self.start = start
        self.end = (end[0]-1, end[1]-1)

        self.__logger = logging.getLogger(__name__)
        # Configure logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler("robot_logs.log"),
                                logging.StreamHandler()
                            ])

    @staticmethod
    def average(num1, num2):
        return round((num1 + num2) / 2)

    def solve_maze(self):
        self.__logger.info(f"start: {self.start}, end: {self.end}")
        self.__logger.info(f"maze shape: {self.maze.shape}")
        self.__logger.info(f"start value: {self.maze[self.start[0]][self.start[1]]}, end value: {self.maze[self.end[0] - 1][self.end[1] - 1]}")

        # Directions for up, down, left, right
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # Queue for BFS
        queue = deque([(self.start, [self.start])])

        # Set for visited nodes
        self.visited = set()

        while queue:
            (x, y), path = queue.popleft()
            if (x, y) not in self.visited:
                if (x, y) == self.end:
                    self.path = path
                    return path
                self.visited.add((x, y))
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < len(self.maze)) and (0 <= ny < len(self.maze[0])) and self.maze[nx][ny] != 1:
                        queue.append(((nx, ny), path + [(nx, ny)]))
        return None  # No path found

    def get_path(self):
        return self.path
