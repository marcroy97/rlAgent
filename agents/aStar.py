import gym
import heapq, random
from gym import wrappers
import time
from testing_environment import build_transition_matrix
import vectostate

# By running python aStar.py, aStar will run on both Taxi and Lunar Lander
# TRIALS changes the number of runs it does

TRIALS = 1000

def main():
	# To run A* on Taxi
	test("Taxi-v2")
	# To run A* on Lunar
	test("LunarLander-v2")


def aStarSearchTaxi(initState, env_):
	"""Search the node that has the lowest combined cost and heuristic first."""
	visited = []
	fringe = PriorityQueue()
	for i in range(len(env_.P[initState])):
		transition = env_.P[initState][i]
		# push (transition, list of actions, cost), priority
		fringe.push((transition[0], [i], -1 * transition[0][2]), -1 * transition[0][2])
	while not fringe.isEmpty():
			state = fringe.pop()
			# see if goal state
			if not state[0][3]:
				if state[0][1] not in visited:
					visited.append(state[0][1])
					for i in range(len(env_.P[state[0][1]])):
						transition = env_.P[state[0][1]][i]
						holder = state[1] + [i]
						fringe.push((transition[0], holder, state[2] + -transition[0][2]), state[2] + -transition[0][2])
			else:
				return (state[1], len(visited))

def aStarSearchLander(initState, env_):
	"""Search the node that has the lowest combined cost and heuristic first."""
	visited = []
	fringe = PriorityQueue()
	counter = 0
	for i in range(4):
		transition = env_[initState][i]
		# push (transition, list of actions, cost), priority
		fringe.push((transition[0], [i], -1 * transition[0][2]), -1 * transition[0][2])
	while not fringe.isEmpty():
			state = fringe.pop()
			# see if goal state
			if not state[0][3]:
				if state[0][1] not in visited:
					visited.append(state[0][1])
					for i in range(len(env_[state[0][1]])):
						transition = env_[state[0][1]][i]
						holder = state[1] + [i]
						fringe.push((transition[0], holder, state[2] + -transition[0][2]), state[2] + -transition[0][2])
			else:
				return (state[1], len(visited))


class PriorityQueue:
	"""
	  Implements a priority queue data structure. Each inserted item
	  has a priority associated with it and the client is usually interested
	  in quick retrieval of the lowest-priority item in the queue. This
	  data structure allows O(1) access to the lowest-priority item.
	"""

	def  __init__(self):
		self.heap = []
		self.count = 0

	def push(self, item, priority):
		entry = (priority, self.count, item)
		heapq.heappush(self.heap, entry)
		self.count += 1

	def pop(self):
		(_, _, item) = heapq.heappop(self.heap)
		return item

	def isEmpty(self):
		return len(self.heap) == 0

	def update(self, item, priority):
		# If item already in priority queue with higher priority, update its priority and rebuild the heap.
		# If item already in priority queue with equal or lower priority, do nothing.
		# If item not in priority queue, do the same thing as self.push.
		for index, (p, c, i) in enumerate(self.heap):
			if i == item:
				if p <= priority:
					break
				del self.heap[index]
				self.heap.append((priority, c, item))
				heapq.heapify(self.heap)
				break
		else:
			self.push(item, priority)


def test(envName, episodes=TRIALS):
		env = gym.make(envName)
		wins = 0
		total_reward = 0
		total_visited = 0
		for _ in range(episodes):
			if envName == 'LunarLander-v2':
				P, actions = build_transition_matrix('LunarLander-v2', env)
				moves, visited = aStarSearchLander(vectostate.vtostate(env.reset()), P)
			elif envName == 'Taxi-v2':
				env_ = env.unwrapped
				moves, visited = aStarSearchTaxi(env.reset(), env_)
			for i in moves:
				next_state, reward, done, info = env.step(i)
				total_reward += reward
			if done:
				wins += 1
			total_visited += visited

		# Print performance statistics 
		if envName == 'LunarLander-v2':
			print("LUNAR LANDER:")
		elif envName == 'Taxi-v2':
			print("TAXI:")
		print("Results after episode number", episodes)
		print("Average reward per episode:", float(total_reward) / episodes)
		print("Average visited per episode:", float(total_visited) / episodes)
		print("Average wins per episode:", float(wins) / float(episodes))


if __name__ == '__main__':
	main()
