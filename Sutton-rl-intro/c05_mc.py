# -*- coding: utf-8 -*-
# @Date    : 2022-01-17
# @Author  : Caster
# @Desc :  online and offline monte carlo algorithm

import numpy as np
from c00_algo import Algo
from c00_env import Env


class MC(Algo):
    def __init__(self, name, env: Env, gamma=1):
        super(MC, self).__init__(name)
        self.env = env
        self.gamma = gamma

    def simulate(self, policy, start_list=None, first_action=None):
        episode = []
        if start_list is not None: # then it should be a list, for simple
            state = self.env.reset(np.random.choice(start_list))
        else:
            state = self.env.reset()

        done = False
        if first_action is not None:
            next_state, reward, done = self.env.step(first_action)
            episode.append((state, first_action, reward, next_state)) # s, a, r, s
            state = next_state
        while not done:
            act = policy(self.env)
            next_state, reward, done = self.env.step(act)
            episode.append((state, act, reward, next_state)) # s, a, r, s
            state = next_state
        return episode


class OnlineMC(MC):
    def __init__(self, env: Env, gamma=1):
        super(OnlineMC, self).__init__('online monte carlo algo', env, gamma)

    def cal_first_visit(self, episode, value, count, isq=False):
        G, length = 0, len(episode)
        R, visit, = [], set()
        for e in reversed(episode): 
            state, reward = e[0], e[2]
            G = self.gamma * G + reward
            R.append(G)
        for i, e in enumerate(episode):
            state, act, reward = e[0], e[1], R[length-i-1]
            if state not in visit:
                if isq:
                    visit.add((state, act))
                    count[state][act] += 1
                    value[state][act] = value[state][act] + (reward - value[state][act])/count[state][act]
                else:
                    visit.add(state)
                    count[state] += 1
                    value[state] = value[state] + (reward - value[state])/count[state]
    
    def cal_every_visit(self, episode, value, count, isq=False):
        G = 0
        for e in episode:
            state, act, reward = e[0], e[1], e[2]
            G = self.gamma * G + reward
            if isq:
                count[state][act] += 1
                value[state][act] = value[state][act] + (G - value[state][act])/count[state][act]
            else:
                count[state] += 1
                value[state] = value[state] + (G - value[state])/count[state]

    def fit_v(self, policy=None, start_list=None, epochs=1000, cal_type='first'):
        states = self.env.get_all_state()
        value = np.zeros(max(states) + 1)
        count = np.zeros(max(states) + 1)

        for epoch in range(1,epochs+1):
            episode = self.simulate(policy, start_list)
            if cal_type == 'first': self.cal_first_visit(episode, value, count)
            else: self.cal_every_visit(episode, value, count)

        return np.round(value, decimals=4)


    def fit_q(self, policy=None, epochs=1000, cal_type='first'):
        """
        in fact, mc can estimate, but...
        """
        raise NotImplementedError("MC ES cat't estimate Q(s, a)")


class OnlineMCES(OnlineMC):
    """
    monte carlo with epsilon greedy strategy to estimate best policy
    """
    def __init__(self, env: Env, gamma=1):
        super(OnlineMCES, self).__init__(env, gamma)

    def fit_v(self, policy=None, epochs=1000, cal_type='first'):
        raise NotImplementedError("MC ES cat't estimate V(s)")

    def fit_q(self, policy=None, epochs=1000, cal_type='first'):
        q_value = {}
        c_value = {}
        state_actions = self.env.get_all_state_action()
        for (s, alist) in state_actions.items():
            q_value[s] = np.zeros(len(alist))
            c_value[s] = np.zeros(len(alist))
        
        def policy(env: Env):
            values = q_value[env._s]
            max_val = np.max(values)
            idx = np.random.choice([act for act, value in enumerate(values) if value == max_val])
            return idx

        s_choices = list(q_value.keys())
        for epoch in range(1, epochs+1):
            s = np.random.choice(s_choices)
            a = np.random.randint(0, len(q_value[s]))
            episode = self.simulate(policy, start_list=[s], first_action=a)
            if cal_type == 'first': self.cal_first_visit(episode, q_value, c_value, isq=True)
            else: self.cal_every_visit(episode, q_value, c_value, isq=True)
        
        for s in q_value.keys():
            q_value[s] = np.round(q_value[s], decimals=4)
        return q_value


class OnlineMCSoft(OnlineMC):
    """
    monte carlo with soft greedy strategy to estimate best policy
    """
    def __init__(self, env: Env, gamma=1, epsi_low=0.1, epsi_high=0.8, decay=1000):
        super(OnlineMCSoft, self).__init__(env, gamma)
        self.epsilon = 0
        self.epsi_low = epsi_low
        self.epsi_high = epsi_high
        self.decay = decay

    def fit_v(self, policy=None, epochs=1000, cal_type='first'):
        raise NotImplementedError("MC Soft cat't estimate V(s)")

    def fit_q(self, policy=None, epochs=1000, cal_type='first'):
        q_value = {}
        c_value = {}
        state_actions = self.env.get_all_state_action()
        for (s, alist) in state_actions.items():
            q_value[s] = np.zeros(len(alist))
            c_value[s] = np.zeros(len(alist))
        
        policy = self.getEpsilonGreedy(q_value)

        for epoch in range(1, epochs+1):
            self.epsilon = self.epsi_low + (self.epsi_high-self.epsi_low) * (np.exp(-1.0 * epoch/self.decay))
            episode = self.simulate(policy)
            if cal_type == 'first': self.cal_first_visit(episode, q_value, c_value, isq=True)
            else: self.cal_every_visit(episode, q_value, c_value, isq=True)
        
        for s in q_value.keys():
            q_value[s] = np.round(q_value[s], decimals=4)
        return q_value


class OfflineMC(MC):
    def __init__(self, env: Env, gamma=1, epsilon=0.05):
        super(OfflineMC, self).__init__('offline monte carlo algo', env, gamma)
        self.epsilon = epsilon

    def simulate(self, policy):
        """
        @param policy: behavior policy func, return value is action with action probility: (act, prob)
        """
        episode = []
        state = self.env.reset()
        done = False
        while not done:
            act, prob = policy(self.env)
            next_state, reward, done = self.env.step(act)
            episode.append((state, act, reward, next_state, prob)) # s, a, r, s, p
            state = next_state
        return episode

    def fit_v(self, policy=None, epochs=1000, cal_type='first'):
        raise NotImplementedError("Offline MC cat't estimate V(s)")

    def fit_q(self, policy=None, epochs=1000, cal_method='weighted'):
        """
        @ param policy: if policy is none, return optimal Q, else return policy Q
        @ param cal_method: 'common' or 'weighted
        """
        q_value = {}
        c_value = {}
        state_actions = self.env.get_all_state_action()
        for (s, alist) in state_actions.items():
            q_value[s] = np.zeros(len(alist))
            c_value[s] = np.zeros(len(alist))
        
        def best_policy(env: Env):
            idx = np.argmax(q_value[env._s])
            return idx
        
        if policy is None:
            policy = best_policy

        def behavior_policy(env: Env): # add random on target policy
            act = policy(env)
            prob = 1 - self.epsilon + self.epsilon/len(q_value[env._s])
            if np.random.random() < self.epsilon: 
                idx = np.random.randint( 0, len(q_value[env._s]))
                if idx != act:
                    prob = self.epsilon/len(q_value[env._s])
                    act = idx
            return act, prob

        for epoch in range(1, epochs+1):
            episode = self.simulate(behavior_policy)
            G, W = 0, 1
            for e in reversed(episode):
                state, act, reward, b_prob = e[0], e[1], e[2], e[4]
                G = self.gamma * G + reward
                if cal_method == 'weighted':
                    c_value[state][act] += W
                    q_value[state][act] = q_value[state][act] + (G - q_value[state][act]) * W/c_value[state][act]
                else: # 'common'
                    c_value[state][act] += 1
                    q_value[state][act] = q_value[state][act] + (G * W - q_value[state][act])/c_value[state][act]
                t_prob = 1 if b_prob > 0.5 else 0
                W = W * t_prob / b_prob 
                if W == 0: break
        
        for s in q_value.keys():
            q_value[s] = np.round(q_value[s], decimals=4)
        return q_value