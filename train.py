import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from collections import deque
from udrlnd.dqn_agent import Agent
from peel import Peel
from unityagents import UnityEnvironment


def print_update(episode, score, duration, newline=False):
    end = None if newline else ""
    print(
        '\rEpisode {}\tScore: {:.2f}\t Duration: {:.2f}'.format(episode, score, duration),
        end=end
    )


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
    """
    DQN Code from Udacity Deep Reinforcement Learning Course
    See udrlnd/LICENSE.md for full MIT license.

    Slightly Modified by Ian Danforth

    Deep Q-Learning.

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
            action = agent.act(state, eps)
            action = int(action)
            next_state, reward, done, _ = env.step(action)
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
    env = Peel(unity_env)

    ########################
    # Agent
    agent = Agent(
        state_size=env.state_size,
        action_size=env.action_size,
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
