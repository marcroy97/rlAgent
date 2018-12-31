import random
from copy import deepcopy
import gym
import numpy as np
import vectostate
import testing_environment
import time

# Class to abstract Q-learning. Takes in as parameters list of actions, list of 
# states, alpha, gammma and epsilon. 
class QLearning:
    def __init__(self, actions, states, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        actions is the discrete action space of the environment
        states is the discrete observation space of the environment
        alpha is learning rate
        gamma is discount factor 
        epsilon is exploration probability 
        current state, action values stored in self.q_table
        """
        self.q_table = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions
        self.states = states

    # Get current Q value from table
    def retrieveQval(self, state, action):
        return self.q_table.get((state, action), 0.0)

    # Apply an update step to the Q value of state, action
    def updateQval(self, state, action, reward, value):
        prev = self.q_table.get((state, action), None)
        if prev is None:
            self.q_table[(state, action)] = reward
        else:
            self.q_table[(state, action)] = prev + self.alpha * (value - prev)

    # Choose the next action according to epsilon greedy policy
    def getAction(self, state):
        # Exploration 
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        # Exploitation
        else:
            qs = [self.retrieveQval(state, act) for act in self.actions]
            mq = max(qs)
            ctr = qs.count(mq)
            if ctr >= 2:
                new = [i for i in range(len(self.actions)) if qs[i] == mq]
                i = random.choice(new)
            else:
                i = qs.index(mq)

            action = self.actions[i]
        # Gradually reduce epsilon 
        self.epsilon = float(self.epsilon ** 1.01)
        return action

    # Parameters to pass into updateQval
    def acquireQval(self, old_state, old_action, reward, new_state):
        update = max([self.retrieveQval(new_state, a) for a in self.actions])
        self.updateQval(old_state, old_action, reward, reward + self.gamma * update)

# Class to abstract the SARSA algorithm
class Sarsa_Learning:
    def __init__(self, actions, states, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        actions is the discrete action space of the environment
        states is the discrete observation space of the environment
        alpha is learning rate
        gamma is discount factor 
        epsilon is exploration probability
        current state, action values stored in self.q_table
        """
        self.q_table = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions
        self.states = states
    
    # Get current Q value from table
    def retrieveQval(self, state, action):
        return self.q_table.get((state, action), 0.0)

    # Apply an update step to the Q value of state, action
    # Initialize to reward if uninitialized
    def updateQval(self, state, action, reward, value):
        prev = self.q_table.get((state, action), None)
        if prev is None:
            self.q_table[(state, action)] = reward 
        else:
            self.q_table[(state, action)] = prev + self.alpha * (value - prev)

    # Choose the next action according to epsilon greedy policy
    def getAction(self, state):
        # Exploration 
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        # Exploitation
        else:
            qs = [self.retrieveQval(state, a) for a in self.actions]
            mq = max(qs)
            ctr = qs.count(mq)
            if ctr >= 2:
                new = [i for i in range(len(self.actions)) if qs[i] == mq]
                i = random.choice(new)
            else:
                i = qs.index(mq)

            action = self.actions[i]
        # Gradually reduce epsilon
        self.epsilon = float(self.epsilon ** 1.01)
        return action

    # Parameters to pass into updateQval
    def acquireQval(self, curr_state, curr_action, reward, next_state, next_action):
        val_of_next = self.retrieveQval(next_state, next_action)
        self.updateQval(curr_state, curr_action, reward, reward + self.gamma * val_of_next)

# Game Agent class. Takes in the game name, the learning algorithm, and the convergence limit c.
class Game_Agent:

    def __init__(self, game, learner, c):
        """
        game is the name of the game
        learner is either q or s for Q_learning or SARSA
        c is the convergence bound
        In the case of the game being LunarLander, we must 
            create the discrete state space. 
        """
        self.env = gym.make(game)
        self.game = game
        self.learner_name = learner
        if game == "LunarLander-v2":
            self.obs_space_n = vectostate.size
        elif game == "Taxi-v2":
            self.obs_space_n = self.env.env.observation_space.n
        if learner == "q":
            self.learner = QLearning([i for i in range(self.env.env.action_space.n)], [i for i in range(self.obs_space_n)])
        elif learner == "s":
            self.learner = Sarsa_Learning([i for i in range(self.env.env.action_space.n)], [i for i in range(self.obs_space_n)])
        self.action = None
        if game == "LunarLander-v2":
            self.state = testing_environment.discretize_state(self.env.reset())
        elif game == "Taxi-v2":
            self.state = self.env.reset()
        self.done = False
        self.converged = False
        self.c = c

    # Perform a state, action update and advance the current state. In the 
    # case of LunarLander, we must use the discretized version of states.
    def update(self):
        self.action = self.learner.getAction(self.state)
        next_state, reward, done, info = self.env.step(self.action)
        if self.game == "LunarLander-v2":
            next_state = testing_environment.discretize_state(next_state)
        if self.learner_name == "q":
            self.learner.acquireQval(self.state, self.action, reward, next_state)
        elif self.learner_name == "s":
            next_action = self.learner.getAction(next_state)
            self.learner.acquireQval(self.state, self.action, reward, next_state, next_action)
        self.state = next_state
        self.done = done

    # Check if the values used to determine the policy have converged. 
    def test_convergence(self, temp):
        # temp is passed in from training as a copy of the q table before 
        # one game is played
        ctr = 0 
        for k1 in self.learner.q_table.keys():
            t = temp.get(k1, 0.0)
            if abs(t - self.learner.retrieveQval(k1[0], k1[1])) > self.c:
                ctr += 1
        if ctr == 0:
            self.converged = True
            print("Converged!")

    # Training function. Until convergence or 1 million iterations, run games
    # resetting at completion and updating state, action pair values. 
    # Also prints out policy, a mapping of state to optimal action(s) that is 
    # stored as self.policy for use in testing.  
    def training(self):
        i = 0
        self.converged = False
        while not self.converged and i < 1000000:
            temp = deepcopy(self.learner.q_table)
            self.state = self.env.reset()
            if self.game == "LunarLander-v2":
                self.state = testing_environment.discretize_state(self.state)
            self.done = False
            while not self.done:
                self.update()
            i += 1
            if i % 100 == 0:
                print("Training Episode:", i)
            self.test_convergence(temp)
            if self.converged:
                print("Converged after step number:", i)
        print("Training completed!")
        print("Constructing policy...")
        self.policy = {}
        for state in self.learner.states:
            val, act = -999999999, []
            for action in self.learner.actions:
                if self.learner.retrieveQval(state, action) > val:
                    val = self.learner.retrieveQval(state, action)
                    act = [action]
                elif self.learner.retrieveQval(state, action) == val:
                    act.append(action)
            self.policy[state] = act
        print('policy:', self.policy)

    # For comparision purposes, run a set of 1000 games making random decisions 
    # and track rewards and timesteps 
    def random_example(self, episodes=1000):
        total_reward, total_steps = 0, 0
        print("Running tests...")
        for i in range(episodes):
            self.state = self.env.reset()
            self.done = False
            while not self.done:
                action = self.env.action_space.sample()
                next_state, reward, done, info = self.env.step(action)
                self.done = done
                self.state = next_state
                total_reward += reward
                total_steps += 1
            if i % 1000 == 0:
                print("Testing phase", i)
        print("Results after game number", episodes, "while using a random strategy")
        print("Average timesteps per game:", float(total_steps / episodes))
        print("Average reward per game:", float(total_reward / episodes))
    
    # Evaluate performance of policy developed in self.training, which is 
    # stored as self.policy 
    def testing(self, episodes=1000):
        total_reward, total_steps = 0, 0
        print("Running tests...")
        for i in range(episodes):
            self.state = self.env.reset()
            if self.game == "LunarLander-v2":
                self.state = testing_environment.discretize_state(self.state)
            self.done = False
            while not self.done:
                action = random.choice(self.policy[self.state])
                next_state, reward, done, info = self.env.step(action)
                self.done = done
                self.state = next_state
                if self.game == "LunarLander-v2":
                    self.state = testing_environment.discretize_state(self.state)
                total_reward += reward
                total_steps += 1
            if i % 1000 == 0:
                print("Testing episode", i)
        print("Results after game number", episodes)
        print("Average timesteps per game:", float(total_steps / episodes))
        print("Average reward per game:", float(total_reward / episodes))
        print("compare to...")
        


# Running the file automatically runs the tests. 
# Also keep track of time to convergence. 
if __name__ == "__main__":
    converge_check = 0.1
    start_time = time.time()
    print("Taxi-v2 using Q-learning")
    taxiq = Game_Agent('Taxi-v2','q', converge_check)
    taxiq.training()
    print("Taxi-v2 using Q-learning took", time.time() - start_time, "seconds to converge or reach 1 million iterations")
    taxiq.testing()
    start_time = time.time()
    print("Taxi-v2 using SARSA")
    taxis = Game_Agent('Taxi-v2','s', converge_check)
    taxis.training()
    print("Taxi-v2 using SARSA took", time.time() - start_time, "seconds to converge or reach 1 million iterations")
    taxis.testing()
    print("Taxi-v2 using a random policy")
    taxis.random_example()
    start_time = time.time()
    print("LunarLander-v2 using Q-learning")
    lunarq = Game_Agent('LunarLander-v2', 'q', converge_check)
    lunarq.training()
    print("LunarLander-v2 using Q-learning took", time.time() - start_time, "seconds to converge or reach 1 million iterations")
    lunarq.testing()
    start_time = time.time()
    print("LunarLander-v2 using SARSA")
    lunars = Game_Agent('LunarLander-v2', 's', converge_check)
    lunars.training()
    print("LunarLander-v2 using SARSA took", time.time() - start_time, "seconds to converge or reach 1 million iterations")
    lunars.testing()
    print("LunarLander-v2 using a random policy")
    lunars.random_example()

   






