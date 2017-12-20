import os
import time
import argparse
from environment import Environment
from agent import Agent


# For python2 compatibility (notice: python2 support will be removed in the future)
try:
    input = raw_input
except NameError:
    pass


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=500, help='Number of training episodes', metavar='')
    parser.add_argument('--n_rows', type=int, default=13, help='Number of rows in the labyrinth', metavar='')
    parser.add_argument('--n_cols', type=int, default=33, help='Number of cols in the labyrinth', metavar='')
    parser.add_argument('--n_walls', type=int, default=100, help='Number of walls in the labyrinth', metavar='')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Agent epsilon for eps-greedy policy', metavar='')
    parser.add_argument('--alpha', type=float, default=1.0, help='Agent alpha for step size', metavar='')
    parser.add_argument('--gamma', type=float, default=1.0, help='Return discount factor', metavar='')
    return parser.parse_args()


if __name__ == '__main__':

    # Parse command line arguments
    args = parse_arguments()

    environment = Environment(n_rows=args.n_rows, n_cols=args.n_cols, n_walls=args.n_walls)

    agent = Agent(n_rows=args.n_rows, n_cols=args.n_cols, epsilon=args.epsilon, alpha=args.alpha, gamma=args.gamma)

    for e in range(args.n_episodes):

        cumulative_reward = 0
        state = environment.start_new_episode()

        while True:

            action = agent.get_action_eps_greedy(*state)
            next_state, reward, is_over = environment.perform_action(action)

            cumulative_reward += reward

            agent.update_Q(old_state=state,
                           action=action,
                           reward=reward,
                           new_state=next_state)

            state = next_state

            if is_over:
                break

        print('Episode: {:03d} - Cumulative reward this episode: {}'.format(e, cumulative_reward))

    input('End of training. \n\nPress `ENTER` to start testing.')

    state = environment.start_new_episode()
    while True:

        os.system('cls' if os.name == 'nt' else 'clear')  # clear screen
        print('Learnt policy:')
        print(environment.policy_str(agent))
        print('Testing policy:')
        print(environment)

        action = agent.get_action_greedy(*state)
        next_state, reward, is_over = environment.perform_action(action)
        state = next_state

        if is_over:
            break

        time.sleep(0.5)
