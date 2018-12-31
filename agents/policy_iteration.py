"""
Solving environment using Policy Itertion and calculate average
score with the derived policy over NUM_TRIALS trials.
USAGE:
    Run policy_iteration within the testing_environment.py file.
    This file simply provides the policy_iteration methods that is called in
    testing_environment.py.
"""
import gym
from gym import wrappers
import numpy as np
from IPython.display import clear_output
from time import sleep
import random

# function copied over from testing_environment.py and modified
def extract_policy_from_V(v, states, actions, P, gamma = 1.0):
    '''Extract the policy given a value-function
    INPUTS:
    - v: the value function we're looking at
    - states: number of states in game space
    - actions: number of possible actions
    - P: the transition table where
        P[s][a] == [(probability, nextstate, reward, done), ...]
    - gamma: the discount factor for rewards
    OUTPUT:
    policy function for discrete state space
    '''
    policy = np.zeros(states)
    # iterate through all of the steps
    for s in range(states):
        # create empty numpy array with length equal to # of actions
        q_sa = np.zeros(actions)
        # iterate through all of the actions
        for a in range(actions):
            for prob, next_state, reward, _  in P[s][a]:
                q_sa[a] += (prob * (reward + gamma * v[next_state]))
        policy[s] = np.argmax(q_sa)
    return policy

def policy_evaluation(policy, states, actions, P, eps, gamma=1.0):
    ''' Part one of Policy Iterations -- Policy Evaluation
    Returns value function from given policy
    INPUTS:
    - env_name: the gym environment
    - policy: policy to be evaluated
    - states: number of states in game space
    - actions: number of possible actions
    - P: the transition table where
        P[s][a] == [(probability, nextstate, reward, done), ...]
    - eps: desired epsilon for convergence
    - gamma: the discount factor for rewards
    OUTPUT:
    - value function for discrete state space
    '''
    # start with all zero numpy array representing value function
    v = np.zeros(states)
    temp = 0
    while True:
        old_v = np.copy(v)
        for s in range(states):
            temp2 = 0
            act = policy[s] # action for state s
            for prob, next_state, reward, _ in P[s][act]:
               temp2 += (prob * (reward + gamma * old_v[next_state]))
            v[s] = temp2
        # if value converged, exit while loop
        if abs(np.sum((np.fabs(old_v - v))) - temp) <= eps:
        # if abs(np.sum((np.fabs(old_v - v))) - temp) < eps:
            break
        temp = np.sum((np.fabs(old_v - v)))
    return v

# modified from medium post
def policy_iteration(states, actions, P, gamma = 1.0, eps = 0.01, max_iterations = 10000):
    '''
    A function that performs policy iteration on the env input.
    INPUTS:
    - env: environment for game space
    - states: number of states in game space
    - actions: number of possible actions
    - P: the transition table where
        P[s][a] == [(probability, nextstate, reward, done), ...]
    - gamma: the discount factor for rewards
    - eps: the desired epsilon
    - max_iterations: limit for # of times we will run policy_iteration
    OUTPUT:
    - policy for discrete states
    '''
    # initialize a random policy
    num_states = len(states)
    policy = np.random.choice(actions, size = num_states)
    for i in range(max_iterations):
        policy_value_func = policy_evaluation(policy, num_states, actions, P, eps, gamma)
        new_policy = extract_policy_from_V(policy_value_func, num_states, actions, P, gamma)
        # if our old policy and our new policy are the same, convergence!
        if np.array_equal(policy, new_policy):
            print ("Policy-Iteration converged at step {}.".format(i+1))
            return policy
        policy = new_policy
    print("Didn't converge even after limit of {} iterations".format(MAX_ITERATION))
    return policy
