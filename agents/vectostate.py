import numpy as np 
import gym 

# We let each value in the observation state vector take on one of two values
# the cuts of which were determined by statistic analysis of the probability 
# distribution of each value
size = 2 ** 8


# Retrieve the actual cuts. Bounds were determined by statistical analyis in 
# testing_dist.py
def get_cuts(states=2):
	x = np.linspace(-1, 1, states)
	y = np.linspace(-0.2, 2, states)
	x_v = np.linspace(-2, 2, states)
	y_v = np.linspace(-2, 0.5, states)
	l_theta = np.linspace(-4, 4, states)
	a_v = np.linspace(-2, 2, states)
	return [x, y, x_v, y_v, l_theta, a_v, [0, 1], [0, 1]]

# Class to abstract the notion of a discrete state
class D_State():

	def __init__(self, v, states):
		"""
		v is the continuous state space vector
		states is the number of discrete states in our representation  
		"""
		self.v = v
		self.cuts = get_cuts(states)
		self.states = states

	# Map the continuous state vector to the closest discrete one
	def mapping(self):
		# Construct a string representation of the vector
		# Currently uses base 2 because that yielded the most practical state space
		# size 
		string = ""
		for i, elt in enumerate(self.v):
			if i < 6:
				dif, index = 2 ** 10, -1
				for n, val in enumerate(self.cuts[i]):
					new = abs(elt - val)
					if new < dif:
						dif = new
						index = n

				string += str(index)
			else:
				if elt == 1:
					string += '1'
				else:
					string += '0'
		
		# Construct the state space number from the string
		# Big endian 
		ret = 0
		for i, c in enumerate(string[::-1]):
			if c == "1":
				ret += self.states ** i
		self.state = ret
		return ret 

# Take in a vector (vec) and return its discrete form
def vtostate(vec):
	c = D_State(vec, 2)
	c.mapping()
	return c.state

# Returns a list of all possible discrete states, which is all numbers from 0 
# to size
def state_list():
	return [i for i in range(size)]

# Example usage
if __name__ == '__main__':
	env = gym.make('LunarLander-v2')
	trials = 1000
	vals = 8
	l = []
	for i in range(2 ** 8):
		l.append(0)

	for j in range(trials):
		done = False
		env.reset()
		while not done:

			# take a random action
			observation, a, done, b = env.step(env.action_space.sample())
			 
			# Convert continuous state observation to discrete state 
			l[vtostate(observation)] += 1

		if j % 100 == 0:
			print("Trial", j, "complete!")

	s = sum(l)
	for i, elt in enumerate(l):
		l[i] = float(elt) / float(s)

	# Statistics about which discrete states were encountered 
	n1 = sum(1 for i in l if i > 0.1)
	n2 = sum(1 for i in l if i == 0)
	print(l)
	print(len(l))
	print(n1)
	print(n2)