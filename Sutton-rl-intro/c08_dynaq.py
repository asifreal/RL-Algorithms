# -*- coding: utf-8 -*-
# @Date    : 2022-01-22
# @Author  : Caster
# @Desc :  dyna-Q

import numpy as np
from c00_algo import Algo
from c00_env import Env
from typing import List, Set, Dict, Tuple, Optional
from c08_model import EnvModel, TimeModel, WeightedModel, TrivialModel
from c08_priori import PriorityModel

class DynaQ(Algo):
    def __init__(self, env: Env, gamma=1, epsi_low=0.1, epsi_high=0.8, decay=1000):
        super(DynaQ, self).__init__('DynaQ Algo')
        self.env = env
        self.gamma = gamma
        self.epsilon = 0
        self.epsi_low = epsi_low
        self.epsi_high = epsi_high
        self.decay = decay
    
    def getModel(self, model) -> EnvModel: 
        if model == 'Weighted':
            return WeightedModel(self.env)
        elif model == 'Time':
            return TimeModel(self.env)
        else:
            return TrivialModel(self.env)

    def fit_v(self, epochs=1000):
        raise NotImplementedError("Dyna Q estimate V(s)")

    def fit_q(self, alpha=0.1, planning_steps=5, steps_alpha=0.01, epochs=1000,  model=None):
        q_value = {}
        state_actions = self.env.get_all_state_action()
        for (s, alist) in state_actions.items():
            q_value[s] = np.zeros(len(alist))
        
        self.model = self.getModel(model)

        def policy(env: Env):
            if np.random.random() < self.epsilon: # random
                idx = np.random.randint( 0, len(q_value[env._s]))
            else:
                values = q_value[env._s]
                max_val = np.max(values)
                idx = np.random.choice([act for act, value in enumerate(values) if value == max_val])
            return idx

        for epoch in range(1, epochs+1):
            self.epsilon = self.epsi_low + (self.epsi_high-self.epsi_low) * (np.exp(-1.0 * epoch/self.decay))
            state = self.env.reset()
            done = False
            while not done:
                act = policy(self.env)
                next_state, reward, done = self.env.step(act)
                #print(state, act, next_state, reward, done)

                # Q-Learning update
                if next_state in q_value:
                    q_value[state][act] += alpha * (reward + self.gamma * np.max(q_value[next_state]) -  q_value[state][act])
                else:
                    q_value[state][act] += alpha * (reward - q_value[state][act])

                # feed the model with experience
                self.model.feed(state, act, next_state, reward)

                # sample experience from the model
                for t in range(planning_steps):
                    s_, a_, ns_, r_ =  self.model.sample()
                    if ns_ in q_value:
                        q_value[s_][a_] += steps_alpha * (r_ + self.gamma * np.max(q_value[ns_]) - q_value[s_][a_])
                    else:
                        q_value[s_][a_] += steps_alpha * (r_  - q_value[s_][a_])

                state = next_state
        
        for s in q_value.keys():
            q_value[s] = np.round(q_value[s], decimals=5)
        return q_value



class PriorityDynaQ(Algo):
    def __init__(self, env: Env, gamma=1,epsi_low=0.1, epsi_high=0.8, decay=1000):
        super(PriorityDynaQ, self).__init__('Priority DynaQ Algo')
        self.env = env
        self.gamma = gamma
        self.epsilon = 0
        self.epsi_low = epsi_low
        self.epsi_high = epsi_high
        self.decay = decay
        self.model = PriorityModel()

    def fit_v(self, epochs=1000):
        raise NotImplementedError("Dyna Q estimate V(s)")

    def fit_q(self, alpha=0.1, theta=0.0001, planning_steps=5, steps_alpha=0.01, epochs=1000):
        q_value = {}
        state_actions = self.env.get_all_state_action()
        for (s, alist) in state_actions.items():
            q_value[s] = np.zeros(len(alist))
        
        def policy(env: Env):
            if np.random.random() < self.epsilon: # random
                idx = np.random.randint( 0, len(q_value[env._s]))
            else:
                values = q_value[env._s]
                max_val = np.max(values)
                idx = np.random.choice([act for act, value in enumerate(values) if value == max_val])
            return idx

        for epoch in range(1, epochs+1):
            self.epsilon = self.epsi_low + (self.epsi_high-self.epsi_low) * (np.exp(-1.0 * epoch/self.decay))
            state = self.env.reset()
            done = False
            while not done:
                act = policy(self.env)
                next_state, reward, done = self.env.step(act)
                #print(state, act, next_state, reward, done)

                # feed the model with experience
                self.model.feed(state, act, next_state, reward)
                # get the priority for current state action pair
                nv = np.max(q_value[next_state]) if next_state in q_value else 0
                priority = np.abs(reward + self.gamma * nv - q_value[state][act])
                if priority > theta:
                    self.model.insert(priority, state, act)
                
                # sample experience from the model
                for t in range(planning_steps):
                    if self.model.empty(): break
                    priority, s_, a_, ns_, r_ = self.model.sample()
                    if ns_ in q_value:
                        q_value[s_][a_] += steps_alpha * (r_ + self.gamma * np.max(q_value[ns_]) - q_value[s_][a_])
                    else:
                        q_value[s_][a_] += steps_alpha * (r_  - q_value[s_][a_])

                    for sp, ap, rp in self.model.predecessor(s_):
                        priority = np.abs(rp + self.gamma * np.max(q_value[s_]) - q_value[sp][ap])
                        if priority > theta:
                            self.model.insert(priority, sp, ap)

                state = next_state
        
        for s in q_value.keys():
            q_value[s] = np.round(q_value[s], decimals=5)
        return q_value