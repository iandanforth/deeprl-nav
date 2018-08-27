import msvcrt  # Windows only
import numpy as np
from unityagents import UnityEnvironment

"""
Run this file to play the bananas environment as a human.

Assumptions:

 - Windows OS
 - You're running this script from a command line

Controls:

After launching the script the Unity window will open. Click back
on the command prompt window used to launch this script so it can capture
keystrokes.

By default the agent will always move forward unless you press a key.

    w - move forward
    a - turn left
    s - move backward
    d - turn right
"""
DEBUG = False


def kbfunc():
    """
    Code from https://stackoverflow.com/a/23098294
    Captures keys from the command prompt window. Windows only.
    """
    x = msvcrt.kbhit()
    if x:
        # getch acquires the character encoded in binary ASCII
        ret = msvcrt.getch()
    else:
        ret = False
    return ret


def print_state(state):
    rays = state[:-2]
    rays = rays.reshape(7, 5)
    r_20, r_90, r_160, r_45, r_135, r_70, r_110 = rays
    print("Banana Wall BadBanana Agent Distance")
    rays = np.array([
        r_20,
        r_45,
        r_70,
        r_90,
        r_110,
        r_135,
        r_160]
    )
    print(rays)
    velocity = state[-2:]
    print(velocity)

def print_ray(episode, score, duration, newline=False):
    end = None if newline else ""
    print(
        '\rEpisode {}\tAverage Score: {:.2f}\t Duration: {:.2f}'.format(episode, score, duration),
        end=end
    )

def main():
    env = UnityEnvironment(file_name="unity/Banana_Windows_x86_64/Banana.exe")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # Debug helpers
    if DEBUG:
        # number of actions
        action_size = brain.vector_action_space_size
        print('Number of actions:', action_size)

        # examine the state space 
        state = env_info.vector_observations[0]
        print('States look like:', state)
        state_size = len(state)
        print('States have length:', state_size)

        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]

    score = 0
    counter = 0
    np.set_printoptions(precision=3, suppress=True)
    while True:
        # Check for keystrokes
        x = kbfunc()
        if x is not False:
            val = x.decode()
            if val == "a":
                action = 2
            elif val == "d":
                action = 3
            elif val == "w":
                action = 0
            elif val == "s":
                action = 1
        else:
            # By default move forward
            action = -1

        counter += 1
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        if DEBUG: print_state(next_state)
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            print("Score: {}".format(score))
            score = 0
            env.reset(train_mode=False)[brain_name]
            print(counter)

    env.close()


if __name__ == '__main__':
    main()
