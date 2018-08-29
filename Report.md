# Implementing a Banana Obsessed Agent

## Summary

 In this project I trained a DQN reinforcement learning agent to reach a score of +13 on 
 average over 100 episodes in the Udacity Deep Reinforcement Learing Nanodegree Bananas 
 environment. (A simplified version of the Banana Collectors Unity-ML environment.) 
 Positive reward is accumulated by running into yellow "good" bananas and avoiding "bad"
 blue bananas which return -1 reward. An episode ends after a fixed interval of 300 steps.

 In addition to adapting provided code to reach this score I contributed two useful components. 
 The first is a [simple wrapper class](peel.py) for the provided Unity environment which makes it 
 directly compatible with the existing class DQN code which was designed for an OpenAI Gym 
 interface.

 The second was to establish human baselines for this environment and propose an alternate measure 
 for declaring this environment "solved" which better measures the ability of an agent.

## Human Level Performance

A human can reach a score of +19 in 24 episodes (likely less) given a proper control scheme. The
average score over 100 additional epsiodes is +16 with a minimum of +12 and a maximum of +22.
All learning was performed "from pixels" as opposed to the below agent which was trained
on non-visual vector observations, thus this is not strictly a bananas-to-bananas comparison.

### Learning

 The `freeplay.py` script provides a human interface to the environment. Using the `wasd` keys
 a user can control the agent and attept to collect reward.

#### Complete Control

 An initial attempt was made to learn how to play the game by directly controlling all aspects of
 the agents movement. Over 20 episodes (roughly 30 minutes) I achieved a maximum score of +12. 
 There were two major limitations in this approach. First, the implementation of my key capture 
 didn't handle switching between long-held keys well. This meant that holding forward and turning 
 left and right as needed (a common play style for first-person movement) was jerky and unreliable. 
 Second, the all-or-nothing (discrete) nature of turns made gameplay more challenging than expected. 
 It should be noted that this limitation is unique to the Udacity Deep Reinforcement Learing 
 Nanodegree Bananas environment and does not appear in the Unity-ML Banana Collectors environment 
 where the action space is continuous.

#### Modified Control

 A slight modification was made to `freeplay.py` to have the default action be `forward` which leaves
 the user free to concentrate on only turning. After only 5 minutes of additional play time 
 (4 episodes) with this new control scheme my max score rose to +19.

 Thus in a total of 24 episodes I was able to greatly exceed the threshold of +13 for "solving" this
 environment. But what about average performance over time?

#### 100 Episode Average was +16 with a minimum of +12

INSERT CHART

 I then played 100 consecutive episodes to determine the performance characteristics of a human 
 player. As can be seen in the above chart there is significant variation between episodes with a 
 min score of +12 (N=2) and a maximum score of +22 (N=2).

 The environment has a random component where the placement of good and bad bananas is stochastic 
 as is the placement and orientation of the player/agent at the start of each episode. This results
 in favorable and unfavorable configurations of the play field for the player. A favorable 
 configuration might have yellow bananas cleanly segregated from blue bananas and in a tight clump
 making them easy to collect quickly. An unfavorable configuration might have bananas evenly
 distributed across the play field with yellow bananas placed close to blue ones. This would
 maximize the travel time required between bananas and increase the chances of accidentally
 collecting a blue banana and thus lowering the total score.

## Plot of Rewards

### "Solved"

INSERT CHART

## Alternate Solution Criteria - Minimum Values



## Methods

 All training was performed on a single Windows 10 desktop machine with an NVIDIA GTX 970. 

### System Setup

|Python   		|3.6.6  	|
|CUDA			|9.0		|
|Pytorch   		|0.4.1   	|
|NVIDIA Driver  |388.13   	|
|Conda			|4.5.10		|

### Learning Algorithm

 The agent uses an implementation of the Deep Q-Learning algorithm with experience replay and a 
 target Q Network. 

 It is important to note that both experience replay and the target Q network
 are useful in the context of neural network function approximization because of the 
 independant and identically distributed (IID) assumption built into stochastic gradient update 
 methods. Without these correlation breaking steps networks tend to be biased toward recent
 experience "forgetting" earlier experience which may still be important and representative for
 the task at hand. 

#### Outer Loop

The agent follows the standard State, Action, Reward, State progression for an off-policy 
reinforcement learning algorithm.

 - The agent selects an action in an epsilon-greedy fashion given the current state and an 
   epsilon value
 - That action is passed to the environment which returns
 	- The next state, the reward, and a "done" signal if the episode is complete.
 - The agent is then passed the tuple of (state, action, reward, next state, done signal) which it uses
   to update its memory and Q networks.
 - Epsilon is then decayed
 - The above is repeated until the environment returns a done signal or the maximum alloted number
   of episodes is reached.

##### DQN Agent

The DQN agent is a Python class which wraps two identical Pytorch neural network models. 
Each Q network was used as provided and has this structure:

Fully Connected Layer (64 units)
		  |
		ReLU
		  |
Fully Connected Layer (64 units)
		  |
		ReLU
		  |
Fully Connected Layer (4 units)

The networks were trained using the {Adam optimizer}(https://pytorch.org/docs/stable/optim.html#torch.optim.Adam).
The learning rate (`LR`) was set to 0.0001. Betas, eps, weight_decay, and amsgrad were all left as 
default.

The first Q network (the "local" network) was used for action selection. The second
Q network was the target network and was used as a more-stable reference during the temporal 
difference (TD) error calculation.

###### Action Selection

The current state was fed to the local Q network to obtain an array of action values. This
was then sampled in an epsilon-greedy fashion. Epsilon was initialized to 1.0 and decayed by 0.995
each episode. A minimum epsilon was set to 0.1 to preserve some exploration even late in training.
(i.e. episode 460 and beyond)

###### Updates and Learning

The agent maintained a replay buffer of up-to 10000 memories i.e. (state, action, reward, 
next_state, done) tuples. Every 4 steps the agent would sample 64 memories randomly from this 
buffer and use those to compute TD errors. A discount factor (`gamma`) of 0.99 was applied during
the calculation of TD errors.

Those errors were then used to compute a mean squared error (MSE) for the batch. This was passed to
Pytorch to calculate error gradients. Finally the network weights were updated by the Adam 
optimizer using those gradients.

After updating the local network (the network from which actions are chosen) a "soft" update was
applied to the target network such that it would track the learning of the local network but at
a greatly reduced rate. The fractional update was controlled by the parameter `tau` which was
set to 0.001.

Note that there was no prioritization of replay samples.





## Ideas for Future Work

### Investigation

As noted above the trained agent demonstrates remarkably poor performance on *some* episodes even
after reaching a relatively high level of average performance. It would be interesting to explore
exactly why some episodes are so challenging to the agent. To do so developing a method to isolate
and "replay" those episodes would be useful.

### Prioritized Replay from Failed Episodes

While the original approach to prioritized replay [CITATION] relied on the TD error to determine
which state transitions were "useful" another metric could be to focus on entire episodes that
were challenging for the agent and prioritize learning from those. Extending this one might explore
if focusing on the extremes of performance, the very good and the very bad and prioritizing those
episodes for learning could help narrow the range of performance across episodes.