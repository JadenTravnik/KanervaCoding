import gym
import numpy as np
import matplotlib.pyplot as plt

from kanerva import BaseKanervaCoder
from agent import QValueAgent


def main():
    env = gym.make("MountainCar-v0")
    n_episodes = 200
    max_steps = 200
    render_rate = 100
    log_rate = 10
    log = {'n_steps': np.zeros(n_episodes), 'td_error': np.zeros(n_episodes)}

    n_features = 2000
    n_closest = 25
    rep = BaseKanervaCoder(env.observation_space, n_features, n_closest)

    agent = QValueAgent(n_features, env.action_space.n, alpha=0.1, epsilon=0.1, gamma=0.99, lmbda=0.99)


    for i_episode in range(n_episodes):
        observation = env.reset()
        old_features = rep.get_features(observation)
        td_error = 0
        for i_step in range(max_steps):
            if i_episode % render_rate == 0:
                env.render()

            action = agent.act(old_features)
            new_observation, reward, done, info = env.step(action)
            new_features = rep.get_features(new_observation)
            tde = agent.update(old_features, action, reward, new_features)
            td_error += tde

            if done:
                agent.erase_traces()
                observation = env.reset()
                if i_episode % log_rate == 0:
                    print(f'Done episode {i_episode} in {i_step+1} steps')
                break
            old_features = new_features

        log['n_steps'][i_episode] = i_step
        log['td_error'][i_episode] = td_error

    env.close()

    fig, axarr = plt.subplots(1,3)

    axarr[0].plot(log['td_error'], color='r')
    axarr[0].title.set_text('TD Error')

    axarr[1].scatter(rep.prototypes[:, 0], rep.prototypes[:, 1], c=rep.visit_counts)
    axarr[1].title.set_text('Prototype Visit Count')

    v = agent.w.mean(axis=1)
    v -= v.min()
    v /= v.max()
    axarr[2].scatter(rep.prototypes[:, 0], rep.prototypes[:, 1], c=v)
    axarr[2].title.set_text('Value Function')

    plt.show()

if __name__ == '__main__':
    main()