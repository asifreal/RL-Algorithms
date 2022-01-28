# -*- coding: utf-8 -*-
# @Date    : 2022-01-28
# @Author  : Caster
# @Desc :  time difference methods

import numpy as np
from c00_algo import Algo
from c00_env import Env


class TD0(Algo):
    def __init__(self, env: Env, gamma=1):
        super(TD0, self).__init__("TDO")
        self.env = env
        self.gamma = gamma

    def fit_v(self, policy=None, epochs=1000, alpha=0.1):
        if policy is None:
            raise NotImplementedError("TD0 cat't estimate V*(s)")

        states = self.env.get_all_state()
        value = np.zeros(max(states) + 1)

        for epoch in range(1,epochs+1):
            state = self.env.reset()
            done = False
            while not done:
                act = policy(self.env)
                next_state, reward, done = self.env.step(act)
                if done:      # V(terminate state) = 0
                    value[state] += alpha * (reward - value[state])
                else:
                    value[state] += alpha * (reward + self.gamma * value[next_state] - value[state])
                state = next_state
        return np.round(value, decimals=4)

    def fit_q(self, policy=None, epochs=1000):
        raise NotImplementedError("TD0 cat't estimate Q(s, a)")


class TD0ForQ(Algo):
    def __init__(self, name, env: Env, gamma=1,epsi_low=0.01, epsi_high=0.8, decay=1000 ):
        super(TD0ForQ, self).__init__(name)
        self.env = env
        self.gamma = gamma
        self.epsi_low = epsi_low
        self.epsi_high = epsi_high
        self.decay = decay

    def fit_v(self, policy=None, epochs=1000, alpha=0.1):
        raise NotImplementedError(f"{self.algo_name} cat't estimate V(s)")

    def fit_q(self, policy=None, epochs=1000, alpha=0.1):
        """
        policy should be None
        """
        q_value = {}
        state_actions = self.env.get_all_state_action()
        for (s, alist) in state_actions.items():
            q_value[s] = np.zeros(len(alist))
        
        policy = self.getEpsilonGreedy(q_value)

        for epoch in range(1, epochs+1):
            self.epsilon = self.epsi_low + (self.epsi_high-self.epsi_low) * (np.exp(-1.0 * epoch/self.decay))
            self.td0_algo(policy, q_value, alpha)

        for s in q_value.keys():
            q_value[s] = np.round(q_value[s], decimals=4)
        return q_value

    def td0_algo(self, policy, q_value, alpha):
        """
        subclass needs to implement this method
        """
        raise NotImplementedError


class Sarsa(TD0ForQ):
    def __init__(self, env: Env, gamma=1,epsi_low=0.01, epsi_high=0.8, decay=1000 ):
        super(Sarsa, self).__init__("sarsa algo", env, gamma, epsi_low, epsi_high, decay)

    def td0_algo(self, policy, q_value, alpha):
        state = self.env.reset()
        act = policy(self.env)
        done = False
        while not done:
            next_state, reward, done = self.env.step(act)
            if not done:
                next_act = policy(self.env)
                q_value[state][act] += alpha * (reward + self.gamma * q_value[next_state][next_act] - q_value[state][act]) 
                state = next_state
                act = next_act
            else:
                q_value[state][act] += alpha * (reward - q_value[state][act]) 


class QLearning(TD0ForQ):
    def __init__(self, env: Env, gamma=1,epsi_low=0.01, epsi_high=0.8, decay=1000 ):
        super(QLearning, self).__init__("Q learning algo", env, gamma, epsi_low, epsi_high, decay)

    def td0_algo(self, policy, q_value, alpha):
        state = self.env.reset()
        done = False
        while not done:
            act = policy(self.env)
            next_state, reward, done = self.env.step(act)
            if not done:
                q_value[state][act] += alpha * (reward + self.gamma * np.max(q_value[next_state]) - q_value[state][act]) 
                state = next_state
            else:
                q_value[state][act] += alpha * (reward - q_value[state][act]) 

class ExpectionSarsa(TD0ForQ):
    def __init__(self, env: Env, gamma=1,epsi_low=0.01, epsi_high=0.8, decay=1000 ):
        super(ExpectionSarsa, self).__init__("expection sarsa", env, gamma, epsi_low, epsi_high, decay)

    def td0_algo(self, policy, q_value, alpha):
        state = self.env.reset()
        done = False
        while not done:
            act = policy(self.env)
            next_state, reward, done = self.env.step(act)
            if not done:
                prob = np.ones_like(q_value[next_state]) * self.epsilon / len(q_value[next_state])
                max_val = np.max(q_value[next_state])
                max_ids = [act for act, value in enumerate(q_value[next_state]) if value == max_val]
                prob[max_ids] = (1 - self.epsilon) / len(max_ids)
                exp_value = np.sum(prob * q_value[next_state])
                q_value[state][act] += alpha * (reward + self.gamma * exp_value - q_value[state][act]) 
                state = next_state
            else:
                q_value[state][act] += alpha * (reward - q_value[state][act]) 


class DoubleQLearning(TD0ForQ):
    def __init__(self, env: Env, gamma=1,epsi_low=0.01, epsi_high=0.8, decay=1000 ):
        super(DoubleQLearning, self).__init__("Double Q learning algo", env, gamma, epsi_low, epsi_high, decay)

    def fit_q(self, policy=None, epochs=1000, alpha=0.1):
        """
        policy should be None
        """
        q_value, q_1_value, q_2_value = {}, {}, {}
        state_actions = self.env.get_all_state_action()
        for (s, alist) in state_actions.items():
            q_1_value[s] = np.zeros(len(alist))
            q_2_value[s] = np.zeros(len(alist))
        
        def policy(env: Env):
            vals = q_1_value[env._s] + q_2_value[env._s]
            if np.random.random() < self.epsilon: # random
                idx = np.random.randint( 0, len(vals))
            else:
                max_val = np.max(vals)
                idx = np.random.choice([act for act, value in enumerate(vals) if value == max_val])
            return idx

        for epoch in range(1, epochs+1):
            self.epsilon = self.epsi_low + (self.epsi_high-self.epsi_low) * (np.exp(-1.0 * epoch/self.decay))
            state = self.env.reset()
            done = False
            while not done:
                act = policy(self.env)
                next_state, reward, done = self.env.step(act)

                if np.random.random() < 0.5: 
                    q1, q2 = q_1_value, q_2_value
                else:
                    q1, q2 = q_2_value, q_1_value
                if not done:
                    q1[state][act] += alpha * (reward + self.gamma * np.max(q2[next_state]) - q1[state][act]) 
                    state = next_state
                else:
                    q1[state][act] += alpha * (reward - q1[state][act]) 

        for s in state_actions.keys():
            q_value[s] = np.round((q_1_value[s] + q_2_value[s]) / 2, decimals=4)
        return q_value