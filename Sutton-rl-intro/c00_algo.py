# -*- coding: utf-8 -*-
# @Date    : 2022-01-17
# @Author  : Caster
# @Desc :  Base class for all reinforcement learning algorithms

import numpy as np
from c00_env import Env

class Algo:
    """
    base algorithm class
    """
    def __init__(self, name='algo'):
        self.algo_name = name

    def fit_q(self):
        """
        fit q values
        """
        raise NotImplementedError

    def fit_v(self):
        """
        fit v values
        """
        raise NotImplementedError

    def getEpsilonGreedy(self, q_value):
        """
        Require setting epsilon in self
        """
        def policy(env: Env):            # epsilon greedy can be put in base class...TBD
            if np.random.random() < self.epsilon: # random
                idx = np.random.randint( 0, len(q_value[env._s]))
            else:
                values = q_value[env._s]
                max_val = np.max(values)
                idx = np.random.choice([act for act, value in enumerate(values) if value == max_val])
            return idx
        return policy