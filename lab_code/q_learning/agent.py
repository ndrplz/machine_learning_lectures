import numpy as np


class Agent:
    def __init__(self, n_rows, n_cols):

        self.n_rows = n_rows
        self.n_cols = n_cols

        self.n_actions = 4

        self.epsilon = 0.01
        self.alpha   = 1
        self.gamma   = 1

        self.Q = np.random.rand(self.n_rows, self.n_cols, self.n_actions)

    def get_action_eps_greedy(self, r, c):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(0, self.n_actions)  # explore
        else:
            action = self.get_action_greedy(r, c)  # exploit
        return action

    def get_action_greedy(self, r, c):
        return np.argmax(self.Q[r, c])

    def update_Q(self, old_state, action, reward, new_state):

        r_old, c_old = old_state
        r_new, c_new = new_state

        action_greedy = self.get_action_greedy(*new_state)
        max_future_return = self.Q[r_new, c_new, action_greedy]
        Q_target = reward + self.gamma * max_future_return

        self.Q[r_old, c_old, action] += self.alpha * (Q_target - self.Q[r_old, c_old, action])

if __name__ == '__main__':
    agent = Agent(n_rows=15, n_cols=15)