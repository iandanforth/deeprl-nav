"""
Run this script to review a previously saved trained agent.
"""
import time
import torch
import argparse
import numpy as np
from random import randint
from dqn_agent import Agent
from unityagents import UnityEnvironment


class SimpleGymWrapper(object):

    def __init__(self, unity_env):
        """
        A thin wrapper for the Bananas Unity Environment provided by
        the Udacity Deep Learning Nanodegree
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


def print_update(episode, score, duration, newline=False):
    end = None if newline else ""
    print(
        '\rEpisode {}\tScore: {:.2f}\t Duration: {:.2f}'.format(
            episode, score, duration
        ),
        end=end
    )


def main():
    #########################
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=5005,
        help="Base port for communication with environment"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for the agent and environment"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint.pth",
        help="Path to a checkpoint file to review"
    )
    parser.add_argument(
        "--episodes",
        default=100,
        type=int,
        help="Number of Episodes to Run as Review"
    )
    args = parser.parse_args()
    base_port = args.port
    seed = randint(0, 1000)
    if args.seed is not None:
        seed = args.seed
    checkpoint = args.checkpoint
    episodes = args.episodes

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
        action_size=env.action_size,
        seed=seed
    )
    agent.qnetwork_local.load_state_dict(torch.load(checkpoint))

    ########################
    # Run
    print("Running {} episodes ...".format(episodes))
    state = env.reset()
    score = 0
    scores = []
    e_start = time.time()
    durations = []
    for i_episode in range(1, episodes + 1):
        state = env.reset()
        score = 0
        e_start = time.time()
        while True:
            action = agent.act(state)
            action = int(action)
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        scores.append(score)
        e_duration = time.time() - e_start
        durations.append(e_duration)

        print_update(i_episode, score, e_duration, newline=False)

    mean_score = np.mean(scores)
    mean_duration = np.mean(durations)
    print()
    print("Review Complete")
    print_update("Final Averages", mean_score, e_duration, newline=True)

if __name__ == '__main__':
    main()
