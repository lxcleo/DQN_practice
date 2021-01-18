import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
#Self-built functions
import DQN
import setup as st
import Experience_replay as ep

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display
# Initialize parameters
batch_size = 256
# Discount factor used in bellman equation
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.0001
# Update target network every 10 epsides
target_update = 10
# Number of frames being stored in the replay memory
memory_size = 10000
# Learning rate
lr = 0.001
num_epsiodes = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = st.EnvManager(device)
strategy = st.Epsilon(eps_start, eps_end, eps_decay)
agent = st.ActionSelection(strategy, em.num_of_actions_avaiable(), device)
memory = ep.ReplayMemory(memory_size)
# Create a tuple to store expereince replay memory
Experience = namedtuple('Experience',('state','action','next_state','reward'))

# Define NN 
policy_net = DQN.DQN(em.get_height(), em.get_width()).to(device)
target_net = DQN.DQN(em.get_height(), em.get_width()).to(device)
# Map each layers with its parameter tesnor using stat_dict
# load_state_dict will load parameters of a NN
target_net.load_state_dict(policy_net.state_dict())
# eval() will let torch know it is not in training mode
target_net.eval()
optimizer = optim.SGD(params = policy_net.parameters(), lr = lr)
episode_duration = []


def extract_tensors(experiences):
	batch = Experience(*zip(*experiences))

	t1 = torch.cat(batch.state)
	t2 = torch.cat(batch.action)
	t3 = torch.cat(batch.reward)
	t4 = torch.cat(batch.next_state)

	return(t1,t2,t3,t4)




for epsiode in range(num_epsiodes):
	em.reset()
	state = em.get_state()

	for timestep in count():
		action = agent.selection(state, policy_net)
		reward = em.take_action(action)
		next_state = em.get_state()
		memory.push(Experience(state, action, next_state, reward))
		state = next_state

		if memory.can_provide_sample(batch_size):
			experiences = memory.sample(batch_size)
			states, actions, rewards, next_states = extract_tensors(experiences)

			current_q_values = st.QValues.get_current(policy_net, states, actions)
			next_q_values = st.QValues.get_next(target_net, states, actions)
			target_q_values = (next_q_values * gamma) + rewards

			loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()


		if em.done:
			episode_duration.append(timestep)
			st.plot(episode_duration, 100)
			break

	if epsiode % target_update == 0:
		target_net.load_state_dict(policy_net.state_dict())

em.close()












 