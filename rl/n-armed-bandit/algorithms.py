import numpy as np

import narmedbandit
import random
import math

def greedy(state: narmedbandit.State):
    return np.argmax(np.array(state.estimate_values))

def e_greedy(epsilon, state: narmedbandit.State):
    if (random.random() < epsilon):
        return random.randint(0, state.action_space_size - 1)
    
    return np.argmax(np.array(state.estimate_values))

def upper_bound_confidence(confidence: float, state: narmedbandit.State):
    return np.argmax(np.array([
        x + confidence * math.sqrt(math.log1p(sum(state.action_count)) / state.action_count[i]) if state.action_count[i] > 0
        else math.inf
        for i, x in enumerate(state.estimate_values)
    ]))