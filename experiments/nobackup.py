"""
Solution Code from Udacity Deep Reinforcement Learning Course

Slightly modified by Ian Danforth
"""
import sys
import os
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from collections import deque
from unityagents import UnityEnvironment

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent)
from dqn_agent import Agent


def print_update(episode, score, duration, newline=False):
    end = None if newline else ""
    print(
        '\rEpisode {}\tScore: {:.2f}\t Duration: {:.2f}'.format(episode, score, duration),
        end=end
    )


def remap_actions(action):
    # 1 -> 2, 2 -> 3
    if action > 0:
        action += 1

    assert action <= 3

    return action


def dqn(
    env,
    agent,
    n_episodes=20000,
    max_t=1000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    solve_score=13,
    use_min=False
):
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
        e_start = time.time()
        for t in range(max_t):
            action = 0
            if t % 5 == 0:
                action = agent.act(state, eps)
                action = int(action)
            env_action = remap_actions(action)
            next_state, reward, done, _ = env.step(env_action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)        # save most recent score for running average
        scores.append(score)               # save most recent score for charting
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        target_score = np.mean(scores_window)
        if use_min:
            target_score = np.min(scores_window)
        e_duration = time.time() - e_start
        print_update(i_episode, target_score, e_duration)
        if i_episode % 100 == 0:
            print_update(i_episode, target_score, e_duration, newline=True)
        if target_score >= solve_score:
            print(
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                    i_episode-100, target_score)
            )
            filename = 'checkpoint-{}.pth'.format(agent.seed)
            torch.save(agent.qnetwork_local.state_dict(), filename)
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
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=5005,
        help="Base port for communication with environment")
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for the agent and environment")
    parser.add_argument(
        "--solve",
        type=float,
        default=13,
        help="The value for the solution criteria"
    )
    parser.add_argument(
        "--mintarget",
        action="store_true",
        help="Use a *minimum* over 100 episodes rather than a mean as the solve critera"
    )
    args = parser.parse_args()
    base_port = args.port
    seed = randint(0, 1000)
    if args.seed is not None:
        seed = args.seed
    use_min = args.mintarget
    solve_score = args.solve

    print("Seed Used: {}".format(seed))

    #########################
    # Environment

    unity_env = UnityEnvironment(
        file_name="unity/Banana_Windows_x86_64/Banana.exe",
        base_port=base_port,
        seed=seed,
        no_graphics=True)
    env = SimpleGymWrapper(unity_env)

    ########################
    # Agent
    agent = Agent(
        state_size=env.state_size,
        action_size=env.action_size - 1,  # No backup
        seed=seed
    )
    scores = dqn(env, agent, solve_score=solve_score, use_min=use_min)

    # plot the scores
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == '__main__':
    main()
