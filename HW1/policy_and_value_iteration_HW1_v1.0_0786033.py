# Spring 2020, IOC 5262 Reinforcement Learning
# HW1: Policy Iteration and Value iteration for MDPs
       
import numpy as np
import gym

def get_rewards_and_transitions_from_env(env):
    # Get state and action space sizes
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Intiailize matrices
    R = np.zeros((num_states, num_actions, num_states))
    P = np.zeros((num_states, num_actions, num_states))

    # Get rewards and transition probabilitites for all transitions from an OpenAI gym environment
    for s in range(num_states):
        for a in range(num_actions):
            for transition in env.env.P[s][a]:
                prob, s_, r, done = transition
                R[s, a, s_] = r
                P[s, a, s_] = prob
            # Normalize transition probabilitites across state + action axes
            P[s, a, :] /= np.sum(P[s, a, :])
    return R, P

def value_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    """        
        Run value iteration (You probably need no more than 30 lines)
        
        Input Arguments
        ----------
            env: 
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for value iteration
            eps: float
                for the termination criterion of value iteration 
        ----------
        
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        
        TODOs
        ----------
            1. Initialize the value function V(s)
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve V(s) using the Bellman optimality operator
            4. Derive the optimal policy using V(s)
        ----------
    """
    num_spaces = env.observation_space.n
    num_actions = env.action_space.n
    
    
    ##### ---------FINISH TODOS HERE----------- #####
    
    #  1. Initialize with a random policy and initial value function
    policy = np.array([env.action_space.sample() for _ in range(num_spaces)])
    value_fnc = np.zeros(num_spaces)
    # 2. Get transition probabilities and reward function from the gym env
    R, P = get_rewards_and_transitions_from_env(env) 
    
    for iteratin in range(max_iterations):
        pre_value_fnc = value_fnc.copy()
        
        #3 . Iterate and improve V(s) using the Bellman optimality operator
        # V*(s) = max_a Summation(s',r){ [P(s',r|s,a)][R(s,a,s') +gamma*V*(s') ] } = max_a Q*(s,a)
        Q = np.einsum('ijk,ijk -> ij', P, R + gamma * policy)
        value_fnc = np.max(Q, axis=1)

        delta_value_func = np.max(np.abs(value_fnc - pre_value_fnc))
        #print("delta_value_func",delta_value_func)
        if  delta_value_func < eps:
            break
    # pi(s) = arg max_a Q
    policy = np.argmax(Q,axis = 1)

    ##### ---------FINISH TODOS HERE----------- #####
    # Return optimal policy    
    return policy

def policy_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    """ 
        Run policy iteration (You probably need no more than 30 lines)
        
        Input Arguments
        ----------
            env: 
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for the policy evalaution in policy iteration
            eps: float
                for the termination criterion of policy evaluation 
        ----------  
        
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        
        TODOs
        ----------
            1. Initialize with a random policy and initial value function
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve the policy
        ----------
    """
    num_spaces = env.observation_space.n
    num_actions = env.action_space.n
    
    
    ##### ---------FINISH TODOS HERE----------- #####

    #  1. Initialize with a random policy and initial value function
    policy = np.array([env.action_space.sample() for _ in range(num_spaces)])
    value_func = np.zeros(num_spaces)
    
    # 2. Get transition probabilities and reward function from the gym env
    R, P = get_rewards_and_transitions_from_env(env) 

    #  3. Iterate and improve the policy
    for iteratin in range(max_iterations):
########################## Policy Evaluation #############################
        pre_policy = policy.copy()
        for j in range(max_iterations):
            pre_value_func = value_func.copy()
            # V(s) = Summation(s',r){ [P(s',r|s,pi(a))][R(s,pi(s),s') + gamma*V*(s')] } = Summation(s',r) Q(s,a)
            Q = np.einsum('ijk,ijk -> ij', P, R + gamma * value_func)
            one_hot_policy = np.zeros((num_spaces,num_actions))
            for k in range(num_spaces):
                one_hot_policy[k,policy[k]] = 1 
            value_func = np.sum(one_hot_policy*Q,1)

            delta_value_func = np.max(np.abs(pre_value_func - value_func))
            if delta_value_func < eps :
                break
########################## Policy Improbement #############################
        Q = np.einsum('ijk,ijk -> ij', P, R + gamma * value_func)
        # pi(s) = arg max_a Q
        policy =  np.argmax(Q, axis=1)
        if np.array_equal(policy, pre_policy):
           break     
    ##### ---------FINISH TODOS HERE----------- #####

    # Return optimal policy
    return policy

def print_policy(policy_name,policy, mapping=None, shape=(0,)):
    print("-"*30,policy_name,"-"*30)
    print(np.array([mapping[action] for action in policy]).reshape(shape))


def run_pi_and_vi(env_name):
    """ 
        Enforce policy iteration and value iteration
    """    
    env = gym.make(env_name)
    env.render()
    print('== {} =='.format(env_name))
    print('# of actions:', env.action_space.n)
    print('# of states:', env.observation_space.n)
    print(env.env.desc)

    vi_policy = value_iteration(env)
    pi_policy = policy_iteration(env)

    return pi_policy, vi_policy



if __name__ == '__main__':
    # OpenAI gym environment: Taxi-v2
    pi_policy, vi_policy = run_pi_and_vi('Taxi-v2')

    # For debugging
    action_map = {0: "S", 1: "N", 2: "E", 3: "W", 4: "P", 5: "D"}
    print_policy("Value Iteration",pi_policy, action_map, shape=None)
    print_policy("Policy Iteration",vi_policy, action_map, shape=None)
    
    # Compare the policies obatined via policy iteration and value iteration
    diff = sum([abs(x-y) for x, y in zip(pi_policy.flatten(), vi_policy.flatten())])        
    print('Discrepancy:', diff)
    



