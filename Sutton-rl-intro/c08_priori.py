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


class PriorityQueue:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = 0

    def add_item(self, item, priority=0):
        if item in self.entry_finder:
            self.remove_item(item)
        entry = [priority, self.counter, item]
        self.counter += 1
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)

    def remove_item(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED

    def pop_item(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        return not self.entry_finder


# Model containing a priority queue for Prioritized Sweeping
class PriorityModel:
    def __init__(self):
        self.model = dict()
        self.priority_queue = PriorityQueue()
        self.predecessors = dict()

    def insert(self, priority, state, action):
        # note the priority queue is a minimum heap, so we use -priority
        self.priority_queue.add_item((state, action), -priority)

    def empty(self):
        return self.priority_queue.empty()

    # get the first item in the priority queue
    def sample(self):
        (state, action), priority = self.priority_queue.pop_item()
        next_state, reward = self.model[(state,action)]
        return -priority, state, action, next_state, reward

    def feed(self, state, action, next_state, reward):
        self.model[(state, action)] = (next_state, reward)
        if next_state not in self.predecessors:
            self.predecessors[next_state] = set()
        self.predecessors[next_state].add((state, action))

    # get all seen predecessors of a state @state
    def predecessor(self, state):
        if state not in self.predecessors:
            return []
        predecessors = []
        for s, a in list(self.predecessors[state]):
            predecessors.append([s, a, self.model[(s, a)][1]])
        return predecessors
