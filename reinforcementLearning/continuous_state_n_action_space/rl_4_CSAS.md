# Reinforcement learning for continuous state and action space

<https://datascience.stackexchange.com/questions/43524/reinforcement-learning-for-continuous-state-and-action-space>

***

## Problem

My goal is to apply Reinforcement Learning to predict the next state of an object under a known 
force in a 3D environment (the approach would be reduced to supervised learning, off-line learning).

### Details of my approach

The current state is the vector representing the position of the object in the environment 
(3 dimensions), and the velocity of the object (3 dimensions). The starting position is randomly 
initialized in the environment, as well as the starting velocity.

The action is the vector representing the movement from state $t$ to state $t+1$.

The reward is just the Euclidean distance between the predicted next state, and the real next 
state (I already have the target position).

What have I done so far?

I have been looking for many methods to do this. Deep Deterministic Policy Gradients works for a 
continuous action space, but in my case I also have a continuous state space. If you are interested 
in this approach, here's the original paper written at DeepMind: 
<http://proceedings.mlr.press/v32/silver14.pdf>

The Actor-Critic approach should work, but it is usually (or always) applied to discrete and 
low-dimensional state space.

Q-Learning and Deep-Q Learning cannot handle high dimensional state space, so my configuration 
would not work even if discretizing the state space.

Inverse Reinforcement Learning (an instance of Imitation learning, with Behavioral Cloning and 
Direct Policy Learning) approximates a reward function when finding the reward function is more 
complicated than finding the policy function. Interesting approach, but I haven't seen any 
implementation, and in my case the reward function is pretty straightforward. Is there a methodology 
to deal with my configuration that I haven't explored?


## Answer

As you say, policy gradient methods work better than value-based methods (like DQN) with continuous 
action spaces.

Your problem seems very similar to the 
[MountainCarContinuous environment](https://github.com/openai/gym/wiki/MountainCarContinuous-v0)
. If you check the 
[leaderboard](https://github.com/openai/gym/wiki/Leaderboard#mountaincarcontinuous-v0), 
you can see how the top score uses A2C. You can find a clear 
implementation of the same algorithm (using the same environment) in the following notebook: 
[Continuous MountainCar Actor Critic Solution.ipynb](https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb).

Here you can also find a collection of 
[Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
. For example, 
[PPO](https://blog.openai.com/openai-baselines-ppo/) 
has proven to be very successful in environments like the one that you describe.

TL;DR Look for Policy gradient methods.

tl;dr Too Long ; Didn't Read 
Literally translates to: That was too long to read. 
Really translates to: I'm too lazy to read the entirety of what you said , but I still want to say 
something.


