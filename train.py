"""
Solution Code from Udacity Deep Reinforcement Learning Course
"""
import numpy as np
import torch
import matplotlib as plt
from collections import deque
from dqn_agent import Agent
from unityagents import UnityEnvironment


def dqn(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)        # save most recent score
        scores.append(score)               # save most recent score
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores


class SimpleGymWrapper(object):

    def __init__(self, unity_env):
        """
        A thin wrapper for the Bananas Unity Environment provided by the Udacity
        Deep Learning Nanodegree
        """
        self.env = unity_env
        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size

        self.initial_state = self.reset()
        self.state_size = len(self.initial_state)

    @staticmethod
    def _get_state(env_info):
        state = env_info.vector_observations[0]
        return state

    @staticmethod
    def _get_reward(env_info):
        reward = env_info.rewards[0]
        return reward

    @staticmethod
    def _get_done(env_info):
        done = env_info.local_done[0]
        return done

    def reset(self, train_mode=True):
        env_info = self.env.reset(train_mode)[self.brain_name]
        return self._get_state(env_info)

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        next_state = self._get_state(env_info)
        reward = self._get_reward(env_info)
        done = self._get_done(env_info)
        return (next_state, reward, done, env_info)


def main():

    #########################
    # Environment

    unity_env = UnityEnvironment(file_name="unity/Banana_Windows_x86_64/Banana.exe")
    env = SimpleGymWrapper(unity_env)

    ########################
    # Agent
    agent = Agent(state_size=env.state_size, action_size=env.action_size, seed=0)
    scores = dqn(env, agent)

    # plot the scores
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == '__main__':
    main()
