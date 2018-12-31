
"""
Framework to solve an environment by plugging in a reinforcement learning algorithm
and calculate the average score over NUM_TRIALS trials.
USAGE:
    Set eps (epsilon) for state value convergence precision parameter.
    Pick ENV_NAME from https://github.com/openai/gym/wiki/Table-of-environments
        - Note: Must be a DISCRETE observation space
    Pick GAMMA, the discount factor.
    Set NUM_TRIALS to be the number of trials to play the game with the
        resultant policy and average the score over.
    Set MAX_ITERATION to be the number of iterations to run for value iteration
        until you return the values even though they haven't converged.
    Run "python3 testing_environment.py"
"""

from topological_value_iteration import topological_order_value_iteration
from topological_value_iteration import value_iteration
from policy_iteration import policy_iteration
from vectostate import vtostate
import numpy as np
import gym
from gym import wrappers
import networkx as nx
from collections import defaultdict
import random
from scipy import stats
import vectostate
import time

# alternative with scientific notation
# EPS = 1e-10
EPS = .0001
ENV_NAME = 'LunarLander-v2'
GAMMA = 0.9
NUM_TRIALS = 1000
MAX_ITERATION = 1000

def main():
    env_name  = ENV_NAME
    gamma = GAMMA
    eps = EPS
    max_iteration = MAX_ITERATION
    env = gym.make(env_name).env
    states = get_states(env_name, env)
    P, actions = build_transition_matrix(env_name, env)
    start = time.time()
    # USAGE: Comment/Uncomment code here to test desired reinforcement learning alg

    # # testing for ** Value Iteration **
    # optimal_v = value_iteration(P, states, actions, gamma, eps, max_iteration);
    # runtime = start - time.time() / 60.0
    # policy = extract_policy(optimal_v, env, states, P, gamma, V=True)
    # policy_score = evaluate_policy(env, env_name, policy, gamma, n=NUM_TRIALS)
    # print('Policy average score for Value Iteration = ', policy_score)

    # # testing for ** Topological Order Value Iteration **
    # optimal_v = topological_order_value_iteration(P, states, actions, gamma, eps, max_iteration);
    # runtime = start - time.time() / 60.0
    # policy = extract_policy(optimal_v, env, states, P, gamma, V=True)
    # policy_score = evaluate_policy(env, env_name, policy, gamma, n=NUM_TRIALS)
    # print('Policy average score for Topological Order Value Iteration = ', policy_score)

    # testing for ** Policy Iteration **
    policy_iter = policy_iteration(states, actions, P, gamma, eps, max_iteration)
    runtime = (time.time() - start) / 60.0
    policy_score = evaluate_policy(env, env_name, policy_iter, gamma, n=NUM_TRIALS)
    print('Policy average score for Policy Iteration = ', policy_score)
    print("Time to converge: {} minutes".format(runtime))

def discretize_state(env_name, s):
    '''
    Accepts continuous state, returns discretized version
    ex: fix the # of states to be like 50, then
    input (1.2342, 8.3452, -2.23941, 4) --> return state x in [0, num_states]
    '''
    if env_name == 'Taxi-v2':
    	return s
    elif env_name == 'LunarLander-v2':
    	return vtostate(s)
    else:
    	raise ValueError("Invalid environment.")

def get_states(env_name, env):
    '''
    Returns a set of states.
    '''
    if env_name == 'Taxi-v2':
        return range(env.unwrapped.nS)
    elif env_name == 'LunarLander-v2':
        return vectostate.state_list()
    else:
        raise ValueError("Invalid environment.")

def build_transition_matrix(env_name, env):
    '''
    INPUT: environment name and list of states. We only need the list of states if the state space
    is continuous.
    OUTPUT: the transition matrix and the number of actions for the environment
    '''
    if env_name == 'Taxi-v2':
        return env.unwrapped.P, env.unwrapped.nA
    elif env_name == 'LunarLander-v2':
        num_actions = env.action_space.n
        return approx_P(env, env_name, get_states(env_name, env), num_actions), num_actions


    else:
        raise ValueError("Invalid environment.")

def extract_policy(V_or_Q, env, states, P, gamma=1.0, V=False):
    if V:
        return extract_policy_from_V(V_or_Q, env, states, P, gamma)
    else:
        return extract_policy_from_Q(V_or_Q, env, states, gamma)

# modified from medium post, don't claim this as my own code
def extract_policy_from_V(v, env, states, P, gamma):
    """ Extract the policy given a value-function """
    env_ = env.unwrapped
    policy = {s:-1 for s in states}
    for s in states:
        q_sa = np.zeros(env_.action_space.n)
        for a in range(env_.action_space.n):
            for next_sr in P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy

def extract_policy_from_Q(q, env, gamma):
    """ Extract the policy given a value-function """
    env_ = env.unwrapped
    policy = np.zeros(env_.nS)
    for s in range(env_.nS):
        q_sa = np.zeros(env_.action_space.n)
        for a in range(env_.action_space.n):
            for next_sr in env_.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy

# modified from medium post, don't claim this as my own code
def run_episode(env, env_name, policy, gamma = 1.0, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs = discretize_state(env_name, obs)
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward

# modified from medium post, don't claim this as my own code
def evaluate_policy(env, env_name, policy, gamma = 1.0,  n = 1000):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
            run_episode(env, env_name, policy, gamma = gamma, render = False)
            for _ in range(n)]
    return np.mean(scores)


'''
Code to approximate the transition table. Actions are deterministic, so it's just a matter of
checking if a certain state a ever transitions to a certain state b, otherwise the probability of
transitioning is 0.
'''

# (heavily) modified from Medium post, don't claim as original
def sample_episode(env, env_name, states, num_actions, P, remaining_actions, render = False):
    """
    Run episode, keep track of transitions made.
    """
    # average number of steps is 90 until complete so 1000 is a precaution
    max_num_steps = 1000
    obs = env.reset()
    current_state = discretize_state(env_name, obs)
    prev = None
    num_steps = 0
    for _ in range(max_num_steps):
        num_steps += 1
        if render:
            env.render()
        # save previous step
        prev = current_state
        action = generate_action(current_state, remaining_actions, num_actions)
        new_obs, reward, done , _ = env.step(action)
        # save new state
        current_state = discretize_state(env_name, new_obs)
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

def generate_action(current_state, remaining_actions, num_actions):
    if not remaining_actions[current_state]:
        return random.randint(0, num_actions - 1)
    else:
        return remaining_actions[current_state].pop()

def approx_P(env, env_name, states, num_actions,  n =1000):
    """ Evaluates
    """
    # P is a table mapping (state, action) --> list of tuples designating probability of landing
    # at each successor state
    P = {s:{x:[] for x in range(num_actions)} for s in states}
    # keep track of remaining actions for each state
    remaining_actions = {s:set(range(num_actions)) for s in states}
    num_steps = []
    for _ in range(n):
        num_steps.append(sample_episode(env, env_name, states, num_actions, P, remaining_actions, render = False))
    print("average number of steps per episode {}".format(np.mean(num_steps)))
    # print(P)
    # reformat P to proper formatting: P[s][a] = [(probability, nextstate, reward, done), ...]
    for s in states:
        for a in range(num_actions):
            # take the most frequent successor if any successors exist
            if P[s][a]:
            	# all states that were ever transitioned to from s
                successors = [tup[0] for tup in P[s][a]]
                # most frequent successor
                successor_state = stats.mode(successors)[0][0]
                done = False
                reward = -1 * np.inf
                for tup in P[s][a]:
                    if tup[0] == successor_state:
                        done = done or tup[2]
                        reward = max(reward, tup[1])
                P[s][a] = [(1.0, successor_state, reward, done)]
    return P

if __name__ == '__main__':
    main()
