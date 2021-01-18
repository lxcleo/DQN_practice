import random
class ReplayMemory():
	# Capacity = N how many frames could be stored
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.count = 0

	def push(self,experience):
		if len(self.memory) < self.capacity:
			self.memory.append(experience)
		else:
			self.memory[self.count % self.capacity] = experience
			self.count += 1

	def can_provide_sample(self, batch_size):
		return len(self.memory) >= batch_size


	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)



