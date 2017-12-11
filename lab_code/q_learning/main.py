import time
from environment import Environment
from agent import Agent


if __name__ == '__main__':

    n_episodes = 1000
    n_rows, n_cols = 13, 33
    n_walls = 150

    environment = Environment(n_rows, n_cols, n_walls)

    agent = Agent(n_rows, n_cols)

    for e in range(n_episodes):
        print(environment.policy_str(agent))

        episode_return = 0
        state = environment.start_new_episode()

        while True:

            action = agent.get_action_eps_greedy(*state)
            next_state, reward, is_over = environment.perform_action(action)

            episode_return += reward  # todo add discount

            agent.update_Q(old_state=state,
                           action=action,
                           reward=reward,
                           new_state=next_state)

            state = next_state

            if is_over:
                break

        print('Episode: {:03d} - Return this episode: {}'.format(e, episode_return))

    # Test
    state = environment.start_new_episode()
    while True:
        print(environment)
        action = agent.get_action_greedy(*state)
        next_state, reward, is_over = environment.perform_action(action)
        state = next_state
        if is_over:
            break
        time.sleep(0.5)
