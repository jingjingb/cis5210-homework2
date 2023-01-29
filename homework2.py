# Include your imports here, if any are used.

import math
import copy

############################################################
# CIS 521: Homework 2
############################################################

student_name = "Jingjing Bai"


############################################################
# Section 1: N-Queens
############################################################

def num_placements_all(n):
    return math.factorial(n*n)/math.factorial(n)/math.factorial(n*n-n)


def num_placements_one_per_row(n):
    return n**n


def n_queens_valid(board):
    # Check same column.
    if len(set(board)) < len(board):
        return False
    # Check if diag.
    cache = {}
    for row in range(len(board)):
        col = board[row]
        # Check if conflicts with existing queens.
        for row_, col_ in cache.items():
            if abs(row - row_) == abs(col - col_):
                return False
        cache[row] = col
    return True


def n_queens_solutions(n):
    solutions = []

    def DFS(path, solutions):
        for i in range(n):
            if i not in path and n_queens_valid(list(path+[i])):
                if len(path) == n-1:
                    solutions.append(list(path + [i]))
                else:
                    DFS(path+[i], solutions)
    DFS([], solutions)
    return solutions


############################################################
# Section 2: Lights Out
############################################################

class LightsOutPuzzle(object):

    def __init__(self, board):
        self.board = board

    def get_board(self):
        return self.board

    def perform_move(self, row, col):
        self.board[row][col] = not self.board[row][col]
        # perform the four neighbors
        for delta_row, delta_col in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            # check boundary conditions
            if row + delta_row >= 0 and \
               row + delta_row < len(self.board):
                if col + delta_col >= 0 and \
                   col + delta_col < len(self.board[0]):
                    self.board[row + delta_row][col + delta_col] = \
                    not self.board[row + delta_row][col + delta_col]

    def scramble(self):
        import random
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                if random.random() < 0.5:
                    self.perform_move(row, col)

    def is_solved(self):
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                if self.board[row][col]:
                    return False
        return True

    def copy(self):
        return copy.deepcopy(self)

    def successors(self):
        successors = {}
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                successor = self.copy()
                successor.perform_move(row, col)
                yield ((row, col), successor)

    def find_solution(self):
        q = [([], self)]
        visited_set = [tuple(tuple(x) for x in self.get_board())]
        while (q):
            moves, curr = q.pop(0)
            for move, next in curr.successors():
                # visit next
                if tuple(tuple(x) for x in next.get_board()) not in visited_set:
                    if next.is_solved():
                        return list(moves + [move])
                    else:
                        visited_set.append(tuple(tuple(x) for x in next.get_board()))
                        q.append((list(moves + [move]), next))
        return None


def create_puzzle(rows, cols):
    return LightsOutPuzzle([[False for _ in range(cols)] for _ in range(rows)])

############################################################
# Section 3: Linear Disk Movement
############################################################

class DiskMovement(object):
    def __init__(self, disks, length, n):
        self.disks = list(disks)
        self.length = length
        self.n = n

    def move(self, from_, to_):
        tmp = list(self.disks)
        disk_to_move = tmp[from_]
        tmp[from_] = 0
        tmp[to_] = disk_to_move
        return DiskMovement(tmp, self.length, self.n)

    def successors(self):
        i = 0
        li = self.disks
        while i < len(self.disks):
            if li[i] != 0:
                if i + 1 < self.length:
                    if li[i + 1] == 0:
                        yield((i, i + 1), self.move(i, i + 1))
                if i + 2 < self.length:
                    if li[i + 2] == 0 and li[i + 1] !=0:
                        yield((i, i + 2), self.move(i, i + 2))
                if i - 1 >= 0:
                    if li[i - 1] == 0:
                        yield((i, i - 1), self.move(i, i - 1))
                if i - 2 >= 0:
                    if li[i - 2] == 0 and li[i - 1] !=0:
                        yield((i, i - 2), self.move(i, i - 2))
            i += 1

def is_solved(dm):
    i = dm.length - 1
    while i >= dm.length - dm.n:
        if dm.disks[i] != 1:
            return False
        i -= 1
    return True

def solve_identical_disks(length, n):
    initial_disks = [1 for i in range(n)]
    for i in range(length - n):
        initial_disks.append(0)
    dm = DiskMovement(initial_disks, length, n)
    moves = {}
    parent = {}
    explored_set = set()
    solution = []
    parent[dm] = dm
    moves[dm] = ()
    q = []
    q.append(dm)
    explored_set.add(tuple(dm.disks))
    if is_solved(dm):
        return moves[dm]
    while len(q)!= 0:
        diskInstance = q.pop(0)
        if is_solved(diskInstance):
            node = diskInstance
            while(parent[node] != node):
                solution.append(moves[node])
                node = parent[node]
            return list(reversed(solution))
        for move, neighbor in diskInstance.successors():
            if tuple(neighbor.disks) not in explored_set:
                parent[neighbor] = diskInstance
                moves[neighbor] = move
                if is_solved(neighbor) is True:
                    node = neighbor
                    while(parent[node] != node):
                        solution.append(moves[node])
                        node = parent[node]
                    return list(reversed(solution))
                explored_set.add(tuple(neighbor.disks))
                q.append(neighbor)
    return None

def is_solved2(dm):
        i = len(dm.disks) - 1
        diskId = 1
        while diskId <= dm.n:
            if dm.disks[i] != diskId:
                return False
            i -= 1
            diskId += 1
        return True

def solve_distinct_disks(length, n):
    initial_disks = [1 for i in range(n)]
    for i in range(length - n):
        initial_disks.append(0)
    dm = DiskMovement(initial_disks, length, n)
    moves = {}
    parent = {}
    explored_set = set()
    solution = []
    parent[dm] = dm
    moves[dm] = ()
    q = []
    q.append(dm)
    explored_set.add(tuple(dm.disks))
    if is_solved2(dm):
        return moves[dm]
    while len(q)!= 0:
        diskInstance = q.pop(0)
        if is_solved(diskInstance):
            node = diskInstance
            while(parent[node] != node):
                solution.append(moves[node])
                node = parent[node]
            return list(reversed(solution))
        for move, neighbor in diskInstance.successors():
            if tuple(neighbor.disks) not in explored_set:
                parent[neighbor] = diskInstance
                moves[neighbor] = move
                if is_solved(neighbor) is True:
                    node = neighbor
                    while(parent[node] != node):
                        solution.append(moves[node])
                        node = parent[node]
                    return list(reversed(solution))
                explored_set.add(tuple(neighbor.disks))
                q.append(neighbor)
    return None

############################################################
# Section 4: Feedback
############################################################

feedback_question_1 = """
1day
"""

feedback_question_2 = """
Question 3
"""

feedback_question_3 = """
order of questions, after Q2, Q3 has clues to find solutions
"""
