import numpy as np


class Agent:
    """
    Class that models a reinforcement learning agent.
    """
    def __init__(self, n_rows, n_cols, epsilon=0.01, alpha=1, gamma=1):

        self.n_rows = n_rows
        self.n_cols = n_cols

        self.n_actions = 4

        self.epsilon = epsilon
        self.alpha   = alpha
        self.gamma   = gamma

        self.Q = np.random.rand(self.n_rows, self.n_cols, self.n_actions)

    def get_action_eps_greedy(self, r, c):
        """
        Epsilon-greedy sampling of next action given the current state.
        
        Parameters
        ----------
        r: int
            Current `y` position in the labyrinth
        c: int
            Current `x` position in the labyrinth

        Returns
        -------
        action: int
            Action sampled according to epsilon-greedy policy.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(0, self.n_actions)  # explore
        else:
            action = self.get_action_greedy(r, c)  # exploit
        return action

    def get_action_greedy(self, r, c):
        """
        Greedy sampling of next action given the current state.

        Parameters
        ----------
        r: int
            Current `y` position in the labyrinth
        c: int
            Current `x` position in the labyrinth

        Returns
        -------
        action: int
            Action sampled according to greedy policy.
        """
        return np.argmax(self.Q[r, c])

    def update_Q(self, old_state, action, reward, new_state):
        """
        Update action-value function Q
        
        Parameters
        ----------
        old_state: tuple
            Previous state of the Environment
        action: int
            Action performed to go from `old_state` to `new_state`
        reward: int
            Reward got after action `action`
        new_state: tuple
            Next state of the Environment

        Returns
        -------
        None
        """
        r_old, c_old = old_state
        r_new, c_new = new_state

        action_greedy = self.get_action_greedy(*new_state)
        max_future_return = self.Q[r_new, c_new, action_greedy]
        Q_target = reward + self.gamma * max_future_return

        self.Q[r_old, c_old, action] += self.alpha * (Q_target - self.Q[r_old, c_old, action])
