import argparse
import gym
import numpy as np 


import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable



parser = argparse.ArgumentParser(description='PyTorch implementation of OpenAi-Gym Atari 2600 Tennis-ram-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()



env = gym.make('Tennis-ram-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

input_layer_size = 128
output_layer_size = 18
hidden_layer_size = 100


class Policy(nn.Module):
	def __init__(self):
		super(Policy, self).__init__()
		# Affine refers to affine transformation, which is a Linear transformation (f(x) = Ax + b) where the term b is 0.
		# In neural networks the b denotes bias.

		# In other words self.affine1 and self.affine2 are the weight matrices between the layers.
		self.affine1 = nn.Linear(input_layer_size, hidden_layer_size)
		self.affine2 = nn.Linear(hidden_layer_size, output_layer_size)

		# Policy gradient needs to store actions and rewards to calculate the backpropagation
		self.saved_actions = []
		self.rewards = []

	def forward(self, x):
		# The input layer is multiplied by the first weight matrix self.affine1. 
		# I.e the weight matrix between the input layer and the hidden layer.
		# The activation function is a Rectifier. ReLu = Rectified Linear unit
		# 
		# One way ReLUs improve neural networks is by speeding up training. The gradient 
		# computation is very simple (either 0 or 1 depending on the sign of x)
		# Logistic and hyperbolic tangent activation functions suffer from vanishing gradient problem,  
		# which makes learning increasingly slower. 
		x = F.relu(self.affine1(x))

		# The hidden layer is multiplied by the second weight matrix self.affine2.
		# The activation function is softmax function, which is a sigmoid function, but 
		# the terms of the vector are normalized so that the sum of the terms of the vector action_scores = 1.
		action_scores = F.softmax(self.affine2(x))
		return action_scores

policy = Policy()
# policy.cuda()
# cuda doesn't work :/
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def select_action(state):
	state = torch.from_numpy(state).float().unsqueeze(0)
	probs = policy(Variable(state))

	# Draw action from the probability distribution of the output. 
	action = probs.multinomial()
	policy.saved_actions.append(action)
	return action.data


def finish_episode():
    R = 0
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for action, r in zip(policy.saved_actions, rewards):
        action.reinforce(r)
    optimizer.zero_grad()
    autograd.backward(policy.saved_actions, [None for _ in policy.saved_actions])
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]



running_reward = 10
for i_episode in range(40):
		# State is the game observation. In this case the ram memory, which is only 1x128x(bytes)
    state = env.reset()
    for t in range(10000): # Don't infinite loop while learning
    		# Forward propagation and action selection:
        action = select_action(state)
        # Step with the action:
        state, reward, done, _ = env.step(action[0,0])
        if args.render:
            env.render()

        policy.rewards.append(reward)
        if done:
            break

    running_reward = running_reward * 0.99 + t * 0.01
    finish_episode()
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            i_episode, t, running_reward))
    
