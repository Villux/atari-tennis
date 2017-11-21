import gym
import math
import random
import datetime
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

# File name
fmt = '%Y%m%d%H%M%S'
stamp = datetime.datetime.now().strftime(fmt)
file_name = 'episode_durations' + stamp + '.p'

env = gym.make('CartPole-v0')

# Disable cuda for now
use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc_output = nn.Linear(128, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden1 = self.relu(self.fc1(x))
        hidden2 = self.relu(self.fc2(hidden1))
        output = self.sigmoid(self.fc_output(hidden2))
        return output

model = Net()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.2
EPS_END = 0.001
EPS_DECAY = 200

optimizer = optim.Adam(model.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    # Epsilon greedy with cool decay
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        return model(Variable(state, volatile=True).type(FloatTensor)).data.max(0)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])


episode_durations = []

last_sync = 0

def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Binary mask
    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

    non_final_states = torch.stack([s for s in batch.next_state if s is not None])
    non_final_next_states_tensor = Variable(non_final_states, volatile=True)

    state_batch = Variable(torch.stack(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - select value by action index
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states_tensor).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # for param in model.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()

# Internet said that 200 is good value for this
max_duration = 0
while max_duration < 200:
    state = torch.from_numpy(np.ascontiguousarray(env.reset(), dtype=np.float32))
    for t in count():
        # Select and perform an action
        action = select_action(state)
        next_state, reward, done, _ = env.step(action[0, 0])
        next_state = torch.from_numpy(np.ascontiguousarray(next_state, dtype=np.float32))

        if done:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, Tensor([reward]))

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            max_duration = t + 1
            episode_durations.append(t + 1)
            pickle.dump(episode_durations, open(file_name, 'wb'))
            break
env.close()
