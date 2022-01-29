# -*- coding: utf-8 -*-
# @Date    : 2022-01-28
# @Author  : Caster
# @Desc :  time difference methods

import numpy as np
from c00_algo import Algo
from c00_env import Env


class TDN(Algo):
    def __init__(self, env: Env, gamma=1, n_steps = 3):
        super(TDN, self).__init__("n-step TD")
        self.env = env
        self.gamma = gamma
        self.n_steps = n_steps

    def fit_v(self, policy=None, epochs=1000, alpha=0.1):
        if policy is None:
            raise NotImplementedError("TD(n) cat't estimate V*(s)")

        states = self.env.get_all_state()
        value = np.zeros(max(states) + 1)

        for epoch in range(1,epochs+1):
            state = self.env.reset()
            done = False
            t, tau, T = 0, 0, np.inf
            sl, rl = [state], []
            while tau < T:
                if t < T:
                    act = policy(self.env)
                    next_state, reward, done = self.env.step(act)
                    rl.append(reward)
                    if done: 
                        T = t + 1
                    else: 
                        sl.append(next_state)
                        state = next_state
                tau = t - self.n_steps + 1
                if tau >= 0 and tau < T:
                    #print(f"t={t}, tau={tau}, T={T}, rl={rl}, sl={sl}")
                    G, gamma = 0, 1
                    for i in range(tau, min(tau + self.n_steps, T)):
                        G += gamma * rl[i]
                        gamma *= self.gamma
                    if tau + self.n_steps < T:
                        G += gamma * value[sl[tau + self.n_steps]]
                    value[sl[tau]] += alpha * (G - value[sl[tau]])
                t+=1
        return np.round(value, decimals=4)

    def fit_q(self, policy=None, epochs=1000):
        raise NotImplementedError("TD(n) cat't estimate Q(s, a)")


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

