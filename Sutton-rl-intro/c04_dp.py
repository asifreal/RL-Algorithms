# -*- coding: utf-8 -*-
# @Date    : 2022-02-06
# @Author  : Caster
# @Desc :  dp

import numpy as np
from c00_algo import Algo
from c00_env import Env


class DP(Algo):
    def __init__(self, env: Env, gamma=1):
        super(DP, self).__init__("Dynamic programming")
        self.env = env
        self.gamma = gamma

    def fit_v(self, policy=None, epochs=10000, epsilon=1e-4):
        states = self.env.get_all_state()
        value = np.zeros(len(states))

        for epoch in range(1,epochs+1):
            v = np.copy(value)
            for s in reversed(range(len(v))):
                t = 0
                for ns, r, p, done in policy.transaction(s, self.env):
                    #print(ns,r,p,done)
                    if done:
                        t += p * (r + 0)
                    else:
                        t += p * (r + self.gamma * v[ns])
                v[s] = t
            if np.sum(np.abs(v - value)) < epsilon:
                value = v
                break
            value = v
            #print(value)
        return np.round(value, decimals=4)

    def fit_q(self, policy=None, epochs=1000, epsilon=1e-4):
        if policy is not None:
            raise NotImplementedError("MC ES cat't estimate Q(s, a)")
        q_value = {}
        states = self.env.get_all_state()
        state_actions = self.env.get_all_state_action()
        value = np.zeros(len(state_actions.keys()))
        for (s, alist) in state_actions.items():
            q_value[s] = np.zeros(len(alist))

        for epoch in range(1, epochs+1):
            v = np.copy(value)
            for s in reversed(q_value.keys()):
                for a in range(len(q_value[s])):
                    t = 0
                    for ns, r, p, done in self.env.possible_result(s, a):
                        if done:
                            t += p * (r + 0)
                        else:
                            t += p * (r + self.gamma * np.max(q_value[ns]))
                    q_value[s][a] = t
            for i, k in enumerate(q_value.keys()):
                value[i] = np.max(q_value[k])
            if np.mean(np.abs(v - value)) < epsilon:
                break
        
        for s in q_value.keys():
            q_value[s] = np.round(q_value[s], decimals=6)
        return q_value