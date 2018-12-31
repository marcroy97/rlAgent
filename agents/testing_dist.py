import numpy as np
import gym
from gym import wrappers
import networkx as nx
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import time

x1s = []
x2s = []
x3s = []
x4s = []
x5s = []
x6s = []
x7s = []
x8s = []
def run_episode(env, gamma = 1.0, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    actions = 4
    obs = env.reset()
    total_reward = 0
    step_idx = 0

    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(random.randint(0, actions - 1))
        x1, x2, x3, x4, x5, x6, x7, x8 = obs
        x1s.append(x1)
        x2s.append(x2)
        x3s.append(x3)
        x4s.append(x4)
        x5s.append(x5)
        x6s.append(x6)
        x7s.append(x7)
        x8s.append(x8)
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward, step_idx

# modified from medium post, don't claim this as my own code
def evaluate(env, gamma = 1.0,  n = 100):
    """ Evaluates 
    """
    vals = [
            run_episode(env, gamma = gamma, render = False)
            for _ in range(n)]
    scores, num_steps = zip(*vals)
    return np.mean(scores), np.mean(num_steps)


env_name  = 'LunarLander-v2'
gamma = .8
env = gym.make(env_name)

start = time.time()
num_trials = 10000
policy_score, avg_num_steps = evaluate(env, gamma, n=num_trials)
runtime = (time.time() - start) / 60
print('Policy average score = {}\n avg num steps = {}'.format(policy_score, avg_num_steps))
print("runtime for {} trials: {} minutes".format(num_trials, runtime))
fig, axs = plt.subplots(4, 2, figsize=(7, 11))
plt.suptitle("Distribution of State Variables", y=.999999999)

axs[0][0].set_title("x position")
axs[0][0].hist(x1s)

axs[0][1].set_title("y position")
axs[0][1].hist(x2s)

axs[1][0].set_title("x velocity")
axs[1][0].hist(x3s)

axs[1][1].set_title("y velocity")
axs[1][1].hist(x4s)

axs[2][0].set_title("lander angle")
axs[2][0].hist(x5s)

axs[2][1].set_title("angular velocity")
axs[2][1].hist(x6s)
# booleans
axs[3][0].set_title("leg 1 - ground contact (boolean)")
axs[3][0].hist(x7s, bins=2)

axs[3][1].set_title("leg 2 - ground contact (boolean)")
axs[3][1].hist(x8s, bins=2)
plt.tight_layout()
plt.savefig('state_dist2.png')