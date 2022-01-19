# -*- coding: utf-8 -*-
# @Date    : 2022-01-17
# @Author  : Caster
# @Desc :  Two Environments for testing Reinforcement learning algos: Gambler and Herman

import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from c00_state import State, LinearState, HermanState

class Env:
    """
    base environment class
    """
    def __init__(self, name='env'):
        self.env_name = name

    def is_done(self) -> bool:
        """
        whether game finish
        """
        raise NotImplementedError

    def set_state(self, state):
        """
        agent can interact with environment use this method
        @param state: the state we want to set
        """
        raise NotImplementedError

    def get_all_state(self):
        """
        return all states available
        @return: List(int)
        """
        raise NotImplementedError

    def reset(self):
        """
        return init state
        @return int
        """
        raise NotImplementedError

    def step(self, action):
        """
        agent can interact with environment use this method
        @param action: what action agent takes
        @return: tuple(nest_state, reward, done)
        """
        raise NotImplementedError


class RandomWalkEnv(Env):
    """
    Simple environmnet for test
    We have N position
    """
    def __init__(self, N):
        super(RandomWalkEnv, self).__init__('random walk env')
        self.N = N

    def step(self, action):
        pass





class GamblerEnv(Env):
    """
    A gambler wants to earn money. he win if having money $ge than N, he lose if no money left.
    whether he win or lose, the game stop finally.
    """
    def __init__(self, N, p, win_reward = 1, lose_reward=0,
                        include_terminate_state=True, seed=None):
        """
        @param N: the target amount of money of the gambler
        @param p: the probability of win each game
        @include_terminate_state: wheter treat last state as a real state, for dp it should be true, for mc it should be false
        """
        super(GamblerEnv, self).__init__('gambler env')
        self.N = N
        self.p = p
        self._s = 0
        self._rw = win_reward
        self._rl = lose_reward
        self._its = include_terminate_state
        self._states : State = LinearState(N, include_terminate_state)
        if seed: np.random.seed(seed)

    def set_state(self, s) -> None:
        assert 0<=s and s<=self.N
        self._s = s

    def get_all_state(self) -> List[int]:
        return self._states.get_all_state()

    def reset(self, val=None) -> int:
        if val is None:
            self._s = np.random.choice(self._states.get_all_state())
        else:
            self.set_state(val)
        return self._s

    def is_done(self) -> bool:
        if self._its:
            return self._s < 0 or self._s > self.N
        else:
            return self._s <=0 or self._s >= self.N

    def step(self, action):
        assert action <= self._s
        ds, reward = 0, 0
        if np.random.random() < self.p: # win
            ds = action
        else: # lose
            ds = -action
        self._s += ds 

        if not self._its:       # reward should in transaction, if no terminal state
            if self._s <= 0: reward = self._rl
            elif self._s >= self.N: reward = self._rw

        return (self._s, reward, self.is_done())


def policy_all_in(env : GamblerEnv):
    mid = int(env.N / 2)
    if env._s < mid:
        return env._s
    else:
        return env.N - env._s

def policy_one_dollar(env : GamblerEnv):
    return 1

def policy_two_dollar(env : GamblerEnv):
    if env._s == 1 or env._s == (env.N - 1):
        return 1
    else:
        return 2

GamblerPolicy = {
    'all_in': policy_all_in,
    'one_dollar': policy_one_dollar,
    'two_dollar': policy_two_dollar
}





# Some Reference
# Probabilistic Verification of Herman’s Self-Stabilisation Algorithm: 
# https://www.prismmodelchecker.org/papers/fac-herman.pdf
# Randomised Self-Stabilising Algorithms: 
# https://www.prismmodelchecker.org/casestudies/self-stabilisation.php
# > In each of the protocols we consider, the network is a ring of identical processes. 
# > The stable configurations are those where there is exactly one process designated as 
# > "privileged" (has a token). This privilege (token) should be passed around the ring forever 
# > in a fair manner.
# > minimum probability of reaching a stable configuration and the maximum expected time
class HermanEnv(Env):
    """
    We have N position with M token.
    But I can't figure out what the policy is in herman, it is just all random
    So I don't think there is an optimal solution in herman...
    """
    def __init__(self, N, M=3, p=0.5, seed=None):
        super(HermanEnv, self).__init__('herman env')
        self.N = N
        self.M = M
        self.p = p
        self._s = 0
        self._states : HermanState = HermanState(N, M)
        if seed: np.random.seed(seed)

    def is_done(self) -> bool:
        return self._s == -1

    def set_state(self, s) -> None:
        assert 0<=s and s< len(self._states.get_all_state())
        self._s = s

    def reset(self, val=None) -> int:
        if val is None:
            self._s = np.random.choice(self._states.get_all_state())
        else:
            self.set_state(val)
        return self._s

    def get_all_state(self) -> List[int]:
        return self._states.get_all_state()

    def step(self, action):
        reward, act = 1, []
        for i in range(self._states.get_state_size(self._s)):
            if np.random.random() < self.p: # win
                act.append(1)
            else: # lose
                act.append(0)
        self._s = self._states.tick(self._s, act)
        return (self._s, reward, self.is_done())

def policy_random(env: HermanEnv):
    return 1

HermanPolicy = {
    'random' : policy_random
}

if __name__ == "__main__":
    env = HermanEnv(5, 3)
    print(env.step(1))