# -*- coding: utf-8 -*-
# @Date    : 2022-01-17
# @Author  : Caster
# @Desc :  States wrapper for Gambler and Herman

import numpy as np
from typing import List, Set, Dict, Tuple, Optional


# maybe useful for merge sym states
class State:
    """
    base state class
    """
    def __init__(self, name='state'):
        self.state_name = name
        
    def get_all_state(self) -> List[int]:
        raise NotImplementedError


class LinearState(State):
    def __init__(self, N, include_terminate_state=True):
        super(LinearState, self).__init__('gambler state')
        self.N = N
        self._its = include_terminate_state
        if self._its:
            self._ss = list(np.arange(N+1))
        else:
            self._ss = list(np.arange(1, N))

    def get_all_state(self):
        return self._ss
