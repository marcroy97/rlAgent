
"""
Functions to solve an environment using Value Iteration and Topological 
Value-Itertion.
"""
import numpy as np
import gym
from gym import wrappers
import networkx as nx
from collections import defaultdict


# only works on discrete state spaces
def build_graph(P, actions, states):
    g = nx.DiGraph()
    # go through each state-action pair, add edge to successor states
    for s in states:
        g.add_node(s)
        for action in range(actions):
            for _, next_s, _, _ in P[s][action]:
                g.add_edge(s, next_s)
    return g, P

def topological_order_value_iteration(P, states, actions, gamma, eps, max_iteration):
    '''
    A function that performs topological value iteration on the MDP input.
    INPUT:
    - P the transition table
        P[s][a] == [(probability, nextstate, reward, done), ...] 
    - states, the discrete sets of states in the environment
    - actions, the discrete set of actions in the environment
    - gamma, the discount factor for rewards
    OUTPUT:
    value function values for discrete states
    '''
    g, P = build_graph(P, actions, states)
    V = defaultdict(int)
    # returned in reversed topological order
    comps = list(nx.kosaraju_strongly_connected_components(g))
    comps.reverse()
    for comp in comps:
        print("component is: {}".format(comp))
        # update the VI values when you add each component
        V.update(hacky_value_iteration(P, comp, actions, V, gamma, eps, max_iteration))
    return V


def hacky_value_iteration(P, states, actions, curr_V, gamma, eps, max_iteration):
    # back up with already computed value iteration values
    V = defaultdict(int, curr_V)
    old_V = defaultdict(int, curr_V)
    for i in range(max_iteration):
        bellman_error = 0
        for s in states:
            old_V[s] = V[s]
            # for each action, take sum over successors via that action
            q_sa = [sum([p*(r + gamma * old_V[s_]) for p, s_, r, _ in P[s][a]]) for a in range(actions)]
            V[s] = max(q_sa)
            residual = abs(V[s] - old_V[s])
            bellman_error = max(bellman_error, residual)
        if bellman_error < eps:
            print("converged after {} iterations".format(i))
            return V
    # set upper bound on number of iterations allowed, because if it doesn't converge, this is problematic
    print("didn't converge but returning anyway at iteration {}".format(max_iteration))
    return V

def value_iteration(P, states, actions, gamma, eps, max_iteration):
    '''
    A function that performs value iteration on the MDP input.
    INPUT:
    - P the transition table
        P[s][a] == [(probability, nextstate, reward, done), ...] 
    - states, the discrete sets of states in the environment
    - actions, the discrete set of actions in the environment
    - gamma, the discount factor for rewards
    OUTPUT:
    value function values for discrete states
    '''
    V = defaultdict(int)
    old_V = defaultdict(int)
    for i in range(max_iteration):
        bellman_error = 0
        for s in states:
            old_V[s] = V[s]
            # for each action, take sum over successors via that action
            q_sa = [sum([p*(r + gamma * old_V[s_]) for p, s_, r, _ in P[s][a]]) for a in range(actions)]
            V[s] = max(q_sa)
            residual = abs(V[s] - old_V[s])
            bellman_error = max(bellman_error, residual)
        if bellman_error < eps:
            print("converged after {} iterations".format(i))
            return V
    # set upper bound on number of iterations allowed, because if it doesn't converge, this is problematic
    print("didn't converge but returning anyway at iteration {}".format(max_iteration))
    return V
