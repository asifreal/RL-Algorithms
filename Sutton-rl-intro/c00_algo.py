# -*- coding: utf-8 -*-
# @Date    : 2022-01-17
# @Author  : Caster
# @Desc :  Base class for all reinforcement learning algorithms

import numpy as np


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