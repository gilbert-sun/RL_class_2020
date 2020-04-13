# Spring 2020, IOC 5262 Reinforcement Learning
# HW1: Monte-Carlo and Temporal-difference policy evaluation

import gym
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from collections import defaultdict

env = gym.make("Blackjack-v0")

def mc_policy_evaluation(policy, env, num_episodes, gamma=1.0):
    """
        Find the value function for a given policy using first-visit Monte-Carlo sampling
        
        Input Arguments
        ----------
            policy: 
                a function that maps a state to action probabilities
            env:
                an OpenAI gym environment
            num_episodes: int
                the number of episodes to sample
            gamma: float
                the discount factor
        ----------
        
        Output
        ----------
            V: dict (that maps from state -> value)
        ----------
    
        TODOs
        ----------
            1. Initialize the value function
            2. Sample an episode and calculate sample returns
            3. Iterate and update the value function
        ----------
        
    """
    
    ##### FINISH TODOS HERE #####

    print("----[mc_policy_evaluation]-1. Initialize the value function")
    V = defaultdict(float)
    N = defaultdict(float)
    returns_sum = defaultdict(float)
    
    for i_episode in range(1,num_episodes+1):
        print("----[mc_policy_evaluation]-2. Sample an episode and calculate sample returns")
        episode = []
        state = env.reset()
        # Generate an episode S0,A0,R1,....,St using pi
        while True:
            action = policy(state)
            next_state, reward, done, infor = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break 
        print("----[mc_policy_evaluation]-3. Entering iterate and update the value function")
        for s,a,r in episode:
            #For firtst time t that state s is visited in episode i 
            # Increment counter of total first visits: N(s) = N(s) +1
            N[s] = N[s]+1
            # Increment total return G(s) = G(s) + Gi,t
            #  Gt= Rt+1 *1 + Rt+2 * gamma + Rt+3 * gamma^2 +...
            for idx,episode_list in enumerate(episode):
                 if episode_list[0] == s:
                     first_idx = idx
                     for idx,episode_list in enumerate(episode[first_idx:]):
                         G = sum([episode_list[2]*(gamma**idx)])
            returns_sum[s] =  returns_sum[s] + G
            # Update estimate V_pi(s) = G(s) / N(s)
            V[s] = returns_sum[s] / N[s]
    print("----[mc_policy_evaluation]- finish")
    ##### FINISH TODOS HERE #####

    return V

def td0_policy_evaluation(policy, env, num_episodes, gamma=1.0):
    """
        Find the value function for the given policy using TD(0)
    
        Input Arguments
        ----------
            policy: 
                a function that maps a state to action probabilities
            env:
                an OpenAI gym environment
            num_episodes: int
                the number of episodes to sample
            gamma: float
                the discount factor
        ----------
    
        Output
        ----------
            V: dict (that maps from state -> value)
        ----------
        
        TODOs
        ----------
            1. Initialize the value function
            2. Sample an episode and calculate TD errors
            3. Iterate and update the value function
        ----------
    """
    ##### FINISH TODOS HERE #####
    alpha = 0.01
    print("----[td0_policy_evaluation]-1. Initialize the value function")
    V = defaultdict(float)
    for i in range(1,num_episodes+1):
        # Initialize S
        state = env.reset()
        
        while True:
            # A <- action given by pi for S
            action = policy(state)
            # Take action A, observe R,S'
            next_state, reward, done, infor = env.step(action)
            # V(S) <- V(S) + alpha[R + gamma*V(S') - V(S)]
            V[state] = V[state] +  alpha * (  reward +  gamma * V[next_state] - V[state] ) 
            # S <- S'
            state = next_state
            # Unitl S is terminal            
            if done:
                break
    #############################

    return V

    

def plot_value_function(V, title="Value Function"):
    """
        Plots the value function as a surface plot.
        (Credit: Denny Britz)
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))
    
    
def apply_policy(observation):
    """
        A policy under which one will stick if the sum of cards is >= 20 and hit otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


if __name__ == '__main__':
    
    #print("observation_space = ",env.observation_space)
    #print("action_space = ",env.action_space.n)
    
    #####################################################################    
    #####States:
    #Players current sum: [0,31]  --- 32 states
    #Dealer's face up card: [1,10]  --- 310 states
    #Whether the player has a usable ace or not: [0,1] ---  2 states
    
    #####Actions:
    # 0 for stick , 1 for hit
    ###########################################################   
    
    V_mc_10k = mc_policy_evaluation(apply_policy, env, num_episodes=10000)
    plot_value_function(V_mc_10k, title="10,000 Steps")

    V_mc_500k = mc_policy_evaluation(apply_policy, env, num_episodes=500000)
    plot_value_function(V_mc_500k, title="500,000 Steps")


    V_td0_10k = td0_policy_evaluation(apply_policy, env, num_episodes=10000)
    plot_value_function(V_td0_10k, title="10,000 Steps")
    V_td0_500k = td0_policy_evaluation(apply_policy, env, num_episodes=500000)
    plot_value_function(V_td0_500k, title="500,000 Steps")
    



