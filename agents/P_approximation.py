'''
Code to approximate the transition table. Actions are deterministic, so it's just a matter of 
checking if a certain state a ever transitions to a certain state b, otherwise the probability of
transitioning is 0.
'''
import numpy as np
import gym
from gym import wrappers
import networkx as nx
from collections import defaultdict
import random
from scipy import stats
from testing_environment import discretize_state


# (heavily) modified from Medium post, don't claim as original
def run_episode(env, states, actions, P, remaining_actions, render = False):
    """ 
    Run episode, keep track of transitions made.
    """
    obs = env.reset()
    current_state = discretize_state(obs)
    prev = None
    num_steps = 0
    while True:
        num_steps += 1
        if render:
            env.render()
        # save previous step
        prev = current_state
        action = generate_action(current_state, remaining_actions)
        new_obs, reward, done , _ = env.step(action)
        # save new state
        current_state = discretize_state(new_obs)
        # note that each action is deterministic, so the current state will always take us to the same
        # successor state. This is complicated becauase when we map continuous states onto a 
        # discrete state space, a certain discrete state will not always map to the same discrete
        # successor, so this is a very poor approximation.
        # In this approximation, I save all successors and later take the majority successor
        # With a deterministic game, must force player to move, so can't allow successor state to be
        # the same state.
        if prev != current_state:
            P[prev][action].append((current_state, reward, done))
        if done:
            break
    return num_steps

def generate_action(current_state, remaining_actions):
    if not remaining_actions[s]:
        return random.randint()
    else:
        return remaining_actions[s].pop()

def approx_P(env, states, num_actions,  n = 1000):
    """ Evaluates 
    """
    # P is a table mapping (state, action) --> list of tuples designating probability of landing
    # at each successor state
    P = {s:{x:[] for x in range(num_actions)} for s in states}
    # keep track of remaining actions for each state
    remaining_actions = {s:set(range(num_actions)) for s in states}
    num_steps = []
    for _ in range(n):
        num_steps.append(run_episode(env, states, P, remaining_actions, render = False))

    print("average number of steps per episode {}".format(np.mean(num_steps)))
    print(P)
    # reformat P to proper formatting: P[s][a] = [(probability, nextstate, reward, done), ...] 
    for s in states:
        for a in range(num_actions):
            # take the most frequent successor if any successors exist
            if P[s][a]:
                successors = [tup[0] for tup in P[s][a]]
                successor_state = stats.mode(successors)[0][0]
                done = False
                reward = -1 * np.inf
                for tup in successors:
                    if tup[0] == successor_state:
                        done = done or tup[2]
                        reward = max(reward, tup[1])
                P[s][a] = [(1.0, successor, reward, done)]
    return P