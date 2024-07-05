from collections import deque


class MazeSolver:

    def __init__(self, maze, start, end):
        self.maze = maze
        self.h, self.w = maze.shape
        self.path = []
        self.start = self.__get_start(start)
        self.end = self.__get_end(end)

    @staticmethod
    def average(num1, num2):
        return round((num1 + num2) / 2)

    def __get_start(self, start):
        return 0, self.average(start[0], start[1])

    def __get_end(self, end):
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

    def get_path(self):
        return self.path
