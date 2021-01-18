import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
	def __init__(self, height, width):
		super(DQN,self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size = 5)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16,32, kernel_size = 5)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32,32, kernel_size = 5)
		self.bn3 = nn.BatchNorm2d(32)

	def forward(self,x):
		x = F.relu(self.bn1(self.conv1))
		x = F.relu(self.bn2(self.conv2))
		x = F.relu(self.bn3(self.conv3))
		# Reshape the output to fit lienar fc layer
		# -1 will like python to calculate the number of 
		# columes according to the given rows
		return x.view(x.size(0), -1)

