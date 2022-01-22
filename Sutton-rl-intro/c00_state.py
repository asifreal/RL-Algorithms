# -*- coding: utf-8 -*-
# @Date    : 2022-01-17
# @Author  : Caster
# @Desc :  States wrapper for Gambler and Herman

import numpy as np
import itertools
from typing import List, Set, Dict, Tuple, Optional
from collections import Counter

# maybe useful for merge sym states
class State(object):
    """
    base state class
    """
    def __init__(self, name='state'):
        self.state_name = name
        
    def get_all_state(self):
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


# refer from https://s-nako.work/2020/04/union-find-in-python/
class UnionFind:
    def __init__(self, size):
        self.rank = [0 for i in range(size)]
        self.parent = [i for i in range(size)]
    
    def find(self, x):
        parent = self.parent[x]
        while parent != x:
            x = parent
            parent = self.parent[x]
        return parent
    
    def union(self, x, y):
        if x == y: return
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        else:
            if self.rank[root_x] >= self.rank[root_y]:
                self.parent[root_y] = root_x
                self.rank[root_x] = max(self.rank[root_y] + 1, self.rank[root_x])
            else:
                self.parent[root_x] = root_y
                self.rank[root_y] = max(self.rank[root_x] + 1, self.rank[root_y])
    def same(self, x, y):
        return self.find(x) == self.find(y)



class HermanState(State):
    def __init__(self, N, M=3):
        super(HermanState, self).__init__('herman state')
        self.N = N
        self.M = M
        assert M % 2 == 1 
        assert N % 2 == 1
        assert M <= N
        self.states_actions_next_state = {}
        self.build_state(N, M)
        self.build_action()

    def build_state(self, N, M):
        """
        find and merge equal states
        if two state become same after rotate or filp, then then shoud be merged
        """
        raw_states = []
        size_dict = {}
        while M >= 3:
            i = 0
            for action in itertools.combinations(range(N), M):
                raw_states.append(tuple(action))
                i+=1
            size_dict[M] = i
            M -= 2

        raw_state_index = {}
        for i, s in enumerate(raw_states):
            raw_state_index[s] = i
        self.raw_states = raw_states
        self.raw_state_index = raw_state_index

        union_find = UnionFind(len(raw_state_index))
        for state in raw_states:
            s = raw_state_index[state]
            if size_dict[len(state)] <= 1: continue
            if union_find.find(s) != s: continue
            symm_state = tuple(sorted((N - x)%N for x in state))
            for i in range(N): 
                # rotate equality
                equal_state = tuple(sorted((x+i)%N for x in state))
                union_find.union(s, raw_state_index[equal_state])
                # symmetric equality
                equal_state = tuple(sorted((x+i)%N for x in symm_state))
                union_find.union(s, raw_state_index[equal_state])
    
        self.shrink_mapping = []       # effectively shrink state space
        for i in range(len(raw_state_index)):
            self.shrink_mapping.append(union_find.find(i))

        self.states = list(set(self.shrink_mapping))
        self.states_mapping = {}
        for i, s in enumerate(self.states):
            self.states_mapping[s] = i
        self._ss = list(range(len(self.states)))

    def build_action(self):
        """
        find and merge equal actions
        if two actions has same effect(result to same next_state), they should be same action
        """
        self.state_action, self._sa = {}, {}
        for s in self._ss:
            act_ns, ns_act = {}, {}
            length = self.get_state_size(s)
            for a in itertools.product([0,1], repeat=length):
                new_s = self.tick(s, a)
                act_ns[a] = new_s
                if new_s in ns_act:
                    ns_act[new_s].append(a)
                else:
                    ns_act[new_s] = [a]
            self.states_actions_next_state[s] = act_ns
            self.state_action[s] = {}    # don't care the action probability in random policy, for Q is for optim
            for i, ns in enumerate(ns_act.keys()):
                self.state_action[s][i] = (ns, ns_act[ns][0])  # shrink (s,a) for Q 
        for k, v in self.state_action.items():
            self._sa[k] = list(v.keys())

    def get_all_state(self):
        return self._ss

    def get_all_state_action(self):
        return self._sa

    def get_state_size(self, state):
        idx = self.states[state]
        return len(self.raw_states[idx])

    def get_state_name(self, state):
        idx = self.states[state]
        return self.raw_states[idx]

    # put tick here for simple
    def tick(self, state, action):
        if state in self.states_actions_next_state and action in self.states_actions_next_state[state]:
            return self.states_actions_next_state[state][action]
        idx = self.states[state]
        state = self.raw_states[idx]
        assert len(state) == len(action)
        sc = Counter([(x+a)%self.N for x, a in zip(state, action)])
        new_state = []
        for k in sc:
            if sc[k] % 2 == 1:
                new_state.append(k)
        new_state = tuple(sorted(new_state))
        if new_state in self.raw_state_index:
            new_raw_idx = self.raw_state_index[new_state]
            new_idx = self.shrink_mapping[new_raw_idx]
            return self.states_mapping[new_idx]
        else:
            return -1     # just let env to handle this terminal state, maybe add one token state is better, but whatever

if __name__ == "__main__":
    hs = HermanState(7, 3) # 35 -> 4
    print(hs.raw_states)
    print(hs.shrink_mapping)
    print(len(hs._ss))
    print(hs._sa)
    print(hs.states)
    print(hs.tick(0, (0,1,1))) # 0-> (0,1,2) -> (0,2,3) -> (0,1,3) -> 1