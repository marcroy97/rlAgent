"""
Framework to solve an environment by plugging in a reinforcement learning algorithm
and calculate the average score over NUM_TRIALS trials.
USAGE:
    Set eps (epsilon) for state value convergence precision parameter.
    Pick GAMMA, the discount factor.
    Set NUM_TRIALS to be the number of trials to play the game with the
        resultant policy and average the score over.
    Set MAX_ITERATION to be the number of iterations to run for value iteration
        until you return the values even though they haven't converged.
    Run "python3 randomized_least_squares.py"
"""

import gym
import numpy as np
from time import sleep
import vectostate
import random
import math


EPS = .1
GAMMA = 0.9
NUM_TRIALS = 1000
MAX_ITERATION = 1000

class RLSVIAgent():

	# Construct the environment, reset the state, and store field variables.
	def __init__(self, env, alpha=0.1, gamma=GAMMA, epsilon=EPS):
		self.env = gym.make(env).env
		if env == "LunarLander-v2":
			obs_space = vectostate.size
		self.q_table = np.zeros([obs_space, self.env.action_space.n])
		self.alpha = alpha
		self.gamma = gamma
		self.weights = [0.5, 0.5]
		self.penalties = 0
		self.state = self.env.reset()
		self.done = False
		self.training = True

	# Take one step
	def performAction(self, action):
		return self.env.step(action)

	# Retrieve the action with the highest current reward in self.qtable.
	def getAction(self, state):
		if self.training:
			cov = np.cov(self.weights)
			w1 = np.random.normal(self.weights[0], cov)
			w2 = np.random.normal(self.weights[1], cov)
			action = None
			holder = -float("Inf")
			for i in range(self.env.action_space.n):
				value = w1 * movingAwayGoal(self.state, action) + w2 * correctOrientation(self.state, action)
				if value >= holder:
					action = i
					holder = value
			return action
		else:
			return np.argmax(self.q_table[vectostate.vtostate(state)])


	# Update the qtable. To be called iteratively in training.
	def updateQValue(self):
		s = vectostate.vtostate(self.state)
		action = self.getAction(s)
		next_state, reward, done, info = self.performAction(action)
		next_s = vectostate.vtostate(next_state)
		self.done = done
		old_value = self.q_table[s, action]
		next_max = np.max(self.q_table[vectostate.vtostate(next_state)])
		new_value = self.weights[0] * movingAwayGoal(self.state, action) + self.weights[1] * correctOrientation(self.state, action)
		for i in range(len(self.weights)):
			new_action = np.argmax(self.q_table[s])
			if i == 0:
				self.weights[i] = self.weights[i] + self.alpha * (reward + self.gamma * self.q_table[next_s, new_action] - new_value) * movingAwayGoal(self.state, action)
			else:
				self.weights[i] = self.weights[i] + self.alpha * (reward + self.gamma * self.q_table[next_s, new_action] - new_value) * correctOrientation(self.state, action)
		self.q_table[s, action] = new_value

		self.state = next_state

	# Run "episodes" training episodes to construct qtable.
	def train(self, episodes=MAX_ITERATION):
		self.training = True
		for i in range(1, episodes):
			counter = 0
			save = self.weights
			self.state = self.env.reset()
			# self.env.render()
			self.done = False
			while not self.done:
				self.updateQValue()
				# self.env.render()
			if self.weights == save:
				counter += 1
				if counter >= 10:
					print("CONVERGED")
					print(i)
			else:
				counter = 0
			if i % 10 == 0:
				# clear_output(wait=True)
				print("Training Episode: ", i)

		print("Training completed!")


	# For testing, we get the next action directly from the qtable.
	def test(self, episodes=NUM_TRIALS):

		total_epochs, total_penalties, total_reward = 0, 0, 0

		for _ in range(episodes):
			self.state = self.env.reset()
			epochs, penalties, reward = 0, 0, 0
			self.done = False
			while not self.done:
				action = np.argmax(self.q_table[vectostate.vtostate(self.state)])
				state, reward, done, info = self.performAction(action)
				self.done = done
				self.state = state
				total_reward += reward

				if reward == -10:
					penalties += 1

				epochs += 1
			print("done!")
			total_penalties += penalties
			total_epochs += epochs

		# Print performance statistics
		print("Results after episode number", episodes)
		print("Average timesteps per episode:", float(total_epochs) / episodes)
		print("Average reward per episode:", float(total_reward) / episodes)
		print("Average penalties per episode:", float(total_penalties) / float(episodes))



def movingAwayGoal(state, action):
	if state[3] > 0 and action == 2:
		return 1
	elif state[2] > 0 and state[0] > 0 and action == 1:
		return 1
	elif state[2] < 0 and state[0] < 0 and action == 3:
		return 1
	else:
		return 0

def correctOrientation(state, action):
	if state[5] < 0.3 and state[5] > -0.3 and action == 0:
		return 1
	if state[5] < 0 and action == 3:
		return 1
	if state[5] > 0 and action == 1:
		return 1
	else:
		return 0


if __name__ == '__main__':
		# Run on Lunar Lander
		print("Randomized Least Squares on LunarLander")
		taxi = RLSVIAgent("LunarLander-v2")
		taxi.train()
		taxi.test()

