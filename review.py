"""
Run this script to review a previously saved trained agent.
"""
import time
import torch
import argparse
import numpy as np
from random import randint
from udrlnd.dqn_agent import Agent
from peel import Peel
from unityagents import UnityEnvironment


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
        "checkpoint",
        help="Path to a checkpoint file to review"
    )
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
        "--episodes",
        default=100,
        type=int,
        help="Number of Episodes to Run as Review"
    )
    parser.add_argument(
        "--graphics",
        default=False,
        action="store_true",
        help="Whether to display visual output during review."
    )
    args = parser.parse_args()
    base_port = args.port
    seed = randint(0, 1000)
    if args.seed is not None:
        seed = args.seed
    checkpoint = args.checkpoint
    episodes = args.episodes
    no_graphics = args.graphics == False

    print("Seed Used: {}".format(seed))

    #########################
    # Environment

    unity_env = UnityEnvironment(
        file_name="unity/Banana_Windows_x86_64/Banana.exe",
        base_port=base_port,
        seed=seed,
        no_graphics=no_graphics)
    env = Peel(unity_env)

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
    print_update("Final Averages", mean_score, mean_duration, newline=True)

if __name__ == '__main__':
    main()
