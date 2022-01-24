# -*- coding: utf-8 -*-
# @Date    : 2022-01-22
# @Author  : Caster
# @Desc :  dyna-Q

import numpy as np
from c00_algo import Algo
from c00_env import Env
from typing import List, Set, Dict, Tuple, Optional
import heapq

class Info:
    def __init__(self, next_state=None, reward=None):
        self.next_state = next_state
        self.reward = reward
        self.count = 0
        self.time = 0

    def setInfo(self, next_state=None, reward=None, time=0):
        self.next_state = next_state
        self.reward = reward
        self.count += 1
        self.time = time
    
    def __str__(self):
        return f"ns={self.next_state}, r={self.reward}, c={self.count}, t={self.time}"


class EnvModel:
    """
    @param env: only for get all state-action paris
    """
    def __init__(self, env:Env):
        self.model : Dict[int, Dict[int, Info]] = dict()
        self.state_visit: Dict[int, int] = dict()
        for s, alist in env.get_all_state_action().items():
            self.model[s] = {i:Info() for i, a in enumerate(alist)}
            self.state_visit[s] = 0

    def feed(self, state, action, next_state, reward):
        raise NotImplementedError

    def sample(self, n=1):
        raise NotImplementedError



# Trivial model for planning in Dyna-Q
class TrivialModel(EnvModel):
    """
    just memory last transition
    """
    def __init__(self, env: Env):
        super(TrivialModel, self).__init__(env)

    def feed(self, state, action, next_state, reward):
        self.model[state][action].setInfo(next_state, reward)
        self.state_visit[state]+=1

    def sample(self, n=1):
        s = np.random.choice([s for s,v in self.state_visit.items() if v > 0])
        a = np.random.choice([a for a,info in self.model[s].items() if info.count > 0])
        info = self.model[s][a]
        return s, a, info.next_state, info.reward


class WeightedModel(TrivialModel):
    def __init__(self, env: Env):
        super(WeightedModel, self).__init__(env)
        self._s = 0

    def sample(self, n=1):
        available_s = [s for s,v in self.state_visit.items() if v > 0]
        if self._s in available_s: s = self._s
        else: s = np.random.choice(available_s)

        available_a = [a for a,info in self.model[s].items() if info.count > 0]
        available_ac = np.array([info.count for a,info in self.model[s].items() if info.count > 0])
        total = np.sum(available_ac)
        a = np.random.choice(available_a, p=available_ac/total)
        info = self.model[s][a]
        self._s = info.next_state
        return s, a, info.next_state, info.reward


# Time-based model for planning in Dyna-Q+
class TimeModel(EnvModel):
    # @timeWeight: also called kappa, the weight for elapsed time in sampling reward, it need to be small
    def __init__(self, env: Env, time_weight=1e-4):
        super(TimeModel, self).__init__(env)
        self.time = 0
        self.time_weight = time_weight

    def feed(self, state, action, next_state, reward):
        self.time += 1
        self.model[state][action].setInfo(next_state, reward, self.time)
        self.state_visit[state]+=1

    def sample(self, n=1):
        s = np.random.choice([s for s,v in self.state_visit.items() if v > 0])
        a = np.random.choice([a for a,info in self.model[s].items() if info.count > 0])
        info = self.model[s][a]

        next_state, reward, time = info.next_state, info.reward, info.time
        # adjust reward with elapsed time since last vist
        reward += self.time_weight * np.sqrt(self.time - time)
        return s, a, next_state, reward