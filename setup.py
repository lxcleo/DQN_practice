import gym
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

class Epsilon():
	def __init__(self,start,end,decay):
		self.start = start
		self.end = end
		self.decay = decay

	def get_EpsiolonRate(self,step):
		rate = self.end + (self.start-self.end) * \
		math.exp(-1. * step * self.decay)
		return rate




class ActionSelection():
	def __init__(self,strategy,num_actions, device):
		self.current_step = 0
		self.strategy = strategy
		self.num_actions = num_actions
		self.device = device
	def selection(self,state,policy_net):
		rate = self.strategy.get_EpsiolonRate(self.current_step)
		self.current_step += 1

		if rate > random.random():
			action = random.randrange(self.num_actions) 
			return torch.tensor([action]).to(self.device)# explore

		else:
			# Do not track on the gradient 
			with torch.no_grad():
				return policy_net(state).argmax(dim=1).to(device) # exploit



class EnvManager():
	def __init__(self,device):
		self.env = gym.make('CartPole-v0').unwrapped
		self.env.reset()
		self.device = device
		self.curretn_screen = None
		self.done = False


	def reset(self):
		self.env.reset()
		self.current_screen = None


	def close(self):
		self.env.close()

	def render(self, mode = 'human'):
		return self.env.render(mode)



	def num_of_actions_avaiable(self):
		return self.env.action_space.n

	def take_action(self):
		_, reward, self.done, _ = self.env.step(action.item())
		return torch.tensor([reward], device = self.device)


	def just_starting(self):
		return self.curretn_screen is None


	def get_processed_screen(self):
		screen = self.render('rgb_array').transpose((2,0,1))
		screen = self.crop_screen(screen)
		return self.transform_screen_data(screen)

	def get_height(self):
		screen = self.get_processed_screen()
		return screen.shape[2]

	def get_width(self):
		screen = self.get_processed_screen()
		return screen.shape[3]

	def crop_screen(self, screen):
		sh = screen.shape[1]
		top = int(sh * 0.4)
		bot = int(sh * 0.8)
		screen = screen[:, top:bot, :]
		return screen

	def get_state(self):
		if self.just_starting() or self.done:
			self.current_screen = self.get_processed_screen()
			black_screen = torch.zeros_like(self.current_screen)
			return black_screen

		else:
			s1 = self.current_screen
			s2 = self.get_processed_screen()
			self.current_screen = s2
			return s2 - s1

	def transform_screen_data(self, screen):
		screen = np.ascontiguousarray(screen, dtype = np.float32) / 255
		screen = torch.from_numpy(screen)

		resize = T.Compose([T.ToPILImage(),
					T.Resize(40, interpolation = Image.CUBIC),
					T.ToTensor()])

		return resize(screen).unsqueeze(0).to(self.device)


def plot(values, moving_avg_period):
	plt.figure(2)
	plt.clf()
	plt.title('Training..')
	plt.xlabel('# of Episode')
	plt.ylabel('Duration')
	plt.plot(values)
	plt.plot(get_moving_average(moving_avg_period, values))
	plt.pause(0.001)
	if is_ipython: display.clear_output(wait=True)




def get_moving_average(period, values):
	values = torch.tensor(values, dtype = torch.float)
	if len(values) >= period:
		moving_avg = values.unfold(dimension = 0, size = period, step = 1) \
		.mean(dim=1).flatten(start_dim=0)
		moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
		return moving_avg.numpy()

	else:
		moving_avg = torch.zeros(len(values))
		return moving_avg.numpy()




