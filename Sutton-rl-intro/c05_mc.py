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

    def simulate(self, policy, start_list):
        episode = []
        if start_list is not None: # then it should be a list, for simple
            state = self.env.reset(np.random.choice(start_list))
        else:
            state = self.env.reset()

        done = False
        while not done:
            act = policy(self.env)
            next_state, reward, done = self.env.step(act)
            episode.append((state, act, reward, next_state)) # s, a, r, s
            state = next_state
        return episode


class OnlineMC(MC):
    def __init__(self, env: Env, gamma=1):
        super(OnlineMC, self).__init__('online monte carlo algo', env, gamma)

    def cal_first_visit(self, episode, value, count):
        G, length = 0, len(episode)
        R, visit, = [], set()
        for e in reversed(episode): 
            state, reward = e[0], e[2]
            G = self.gamma * G + reward
            R.append(G)
        for i, e in enumerate(episode):
            state, reward = e[0], R[length-i-1]
            if state not in visit:
                visit.add(state)
                count[state] += 1
                value[state] = value[state] + (reward - value[state])/count[state]
    
    def cal_every_visit(self, episode, value, count):
        G = 0
        for e in episode:
            state, reward = e[0], e[2]
            G = self.gamma * G + reward
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