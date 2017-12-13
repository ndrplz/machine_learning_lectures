# -*- coding: utf-8 -*-
import random
import numpy as np
from action import Action


class Environment:
    """
    Class the models a reinforcement learning labyrinth environment.
    """
    WALL     = -1
    EMPTY    = 0
    TERMINAL = 2
    AGENT    = 1
    
    def __init__(self, n_rows, n_cols, n_walls):

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_walls = n_walls

        self.entrance_coordinates = (self.n_rows - 1, 0)
        self.exit_coordinates = (0, self.n_cols - 1)

        self.matrix = np.zeros(shape=(self.n_rows, self.n_cols))
        self._init_labyrinth_matrix()

        self.agent_coordinates = self.entrance_coordinates

    def __repr__(self):
        """
        Represent the labyrinth in ASCII-art
        
        Returns
        -------
        environment_description: str
            Text representation of the environment.
        """
        environment_description = ''
        for i in range(0, self.n_rows):
            for j in range(0, self.n_cols):
                if self.agent_coordinates == (i, j):
                    environment_description += '🐐 '
                elif self.matrix[i, j] == Environment.EMPTY:
                    environment_description += '. '
                elif self.matrix[i, j] == Environment.WALL:
                    environment_description += 'x '
                elif self.matrix[i, j] == Environment.TERMINAL:
                    environment_description += '✔ '
            environment_description += '\n'
        return environment_description

    def _init_labyrinth_matrix(self):
        """
        Initialize the labyrinth with the appropriate number of walls.
        """
        self.matrix[self.exit_coordinates] = Environment.TERMINAL

        while self.n_walls != 0:

            i = np.random.randint(1, self.n_rows - 1)
            j = np.random.randint(0, self.n_cols)

            def can_build_wall_here(r, c):
                row_is_odd = r % 2 != 0
                not_already_wall = self.matrix[r, c] != Environment.WALL
                walls_available  = self.n_walls > 0
                still_an_hole = np.sum(self.matrix[r, :] == Environment.WALL) <= self.n_cols - 2
                return not_already_wall and row_is_odd and walls_available and still_an_hole

            if can_build_wall_here(i, j) and random.random() < 0.3:
                self.matrix[i, j] = Environment.WALL
                self.n_walls -= 1
            elif self.matrix[i, j] != Environment.WALL:
                self.matrix[i, j] = Environment.EMPTY

    def policy_str(self, agent):
        """
        Return the string representation of policy learnt by the agent.
        
        Parameters
        ----------
        agent: Agent
            Reinforcement learning agent
            
        Returns
        -------
        policy_description: str
            Text representation of the policy learnt by the agent.
        """
        policy_description = ''
        for i in range(0, self.n_rows):
            for j in range(0, self.n_cols):
                if self.matrix[i, j] == Environment.WALL:
                    policy_description += 'x '
                elif self.matrix[i, j] == Environment.TERMINAL:
                    policy_description += '✔ '
                else:
                    action = agent.get_action_greedy(r=i, c=j)
                    policy_description += Action.to_arrow(action) + ' '
            policy_description += '\n'
        return policy_description

    def exists_cell_at(self, r, c):
        """
        Return True if the cell in (r, c) exists
        """
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def perform_action(self, action):
        """
        Perform an action in the Environment
        
        Parameters
        ----------
        action: Action
            Possible action in [0, 1, 2, 3]
            
        Returns
        -------
        current_state: tuple
            Current state of the Environment
        reward: int
            Reward for action performed
        is_over: bool
            Flag to signal the end of the episode
        """
        (r, c) = self.agent_coordinates
        if action == Action.UP:
            r -= 1
        elif action == Action.DOWN:
            r += 1
        elif action == Action.LEFT:
            c -= 1
        elif action == Action.RIGHT:
            c += 1

        # Check walls and borders
        if self.exists_cell_at(r, c) and self.matrix[r, c] != Environment.WALL:
            self.agent_coordinates = (r, c)

        return self.current_state, self.reward, self.is_over

    def start_new_episode(self):
        """
        Start a new episode in the Environment.
        
        Returns
        -------
        agent_coordinates: tuple
            Agent location at the beginning of the episode
        """
        self.agent_coordinates = self.entrance_coordinates
        return self.agent_coordinates

    @property
    def is_over(self):
        """
        Flag to signal whether the agent reached the exit.
        """
        return self.agent_coordinates == self.exit_coordinates

    @property
    def reward(self):
        """
        Environment reward. Currently is hardcoded to `-1` for each timestep.
        """
        return -1

    @property
    def current_state(self):
        """
        Current state of the Environment.
        """
        return self.agent_coordinates
