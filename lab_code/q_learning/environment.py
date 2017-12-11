# -*- coding: utf-8 -*-
import random
import numpy as np
from agent import Agent
from action import Action


W = WALL = -1
E = EMPTY = 0
T = TERMINAL = 2
M = MUSHROOM = 1


class Environment:

    def __init__(self, n_rows, n_cols, n_walls=30):

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_walls = n_walls

        self.entrance_coordinates = (self.n_rows - 1, 0)
        self.exit_coordinates = (0, self.n_cols - 1)

        self.matrix = np.zeros((self.n_rows, self.n_cols))
        self.init_matrix()

        self.mushroom_coordinates = self.entrance_coordinates

    def __repr__(self):
        strDescription = ""
        for i in range(0, self.n_rows):
            for j in range(0, self.n_cols):
                if self.mushroom_coordinates == (i, j):
                    strDescription += "🐐 "
                elif self.matrix[i, j] == EMPTY:
                    strDescription += ". "
                elif self.matrix[i, j] == WALL:
                    strDescription += "◼︎"[0:3] + " "
                elif self.matrix[i, j] == TERMINAL:
                    strDescription += "✔︎ "
            strDescription += "\n"
        return strDescription

    def init_matrix(self):

        self.matrix[self.exit_coordinates] = TERMINAL

        while self.n_walls != 0:

            i = np.random.randint(1, self.n_rows - 1)
            j = np.random.randint(0, self.n_cols)

            def can_build_wall_here(r, c):
                row_is_odd = r % 2 != 0
                not_already_wall = self.matrix[r, c] != WALL
                walls_available  = self.n_walls > 0
                still_an_hole = np.sum(self.matrix[r, :] == WALL) <= self.n_cols - 2
                return not_already_wall and row_is_odd and walls_available and still_an_hole

            if can_build_wall_here(i, j) and random.random() < 0.3:
                self.matrix[i, j] = WALL
                self.n_walls -= 1
            elif self.matrix[i, j] != WALL:
                self.matrix[i, j] = EMPTY

    def policy_str(self, agent):

        strDescription = ""
        for i in range(0, 10):
            for j in range(0, 10):
                if self.matrix[i, j] == WALL:
                    strDescription += "◼︎"[0:3] + " "
                elif self.matrix[i, j] == TERMINAL:
                    strDescription += "✔︎ "
                else:
                    self.mushroom_coordinates = (i, j)
                    strDescription += str(agent.greedyActionForState(self.current_state)) + " "
            strDescription += "\n"
        return strDescription

    def exists_cell_at(self, r, c):
        """
        Return True if the cell in (r, c) exists
        """
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def perform_action(self, action):

        (r, c) = self.mushroom_coordinates
        if action == Action.UP:
            r -= 1
        elif action == Action.DOWN:
            r += 1
        elif action == Action.LEFT:
            c -= 1
        elif action == Action.RIGHT:
            c += 1

        # Check walls and borders
        if self.exists_cell_at(r, c) and self.matrix[r, c] != WALL:
            self.mushroom_coordinates = (r, c)

        return self.current_state, self.reward, self.is_over

    def start_new_episode(self):
        self.mushroom_coordinates = self.entrance_coordinates

    @property
    def is_over(self):
        return self.mushroom_coordinates == self.exit_coordinates

    @property
    def reward(self):
        return -1

    @property
    def current_state(self):
        return self.mushroom_coordinates


if __name__ == '__main__':

    environment = Environment(15, 15, n_walls=80)
    print(environment)
