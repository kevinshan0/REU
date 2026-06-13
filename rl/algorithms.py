import narmedbandit
import random

def greedy(state: narmedbandit.State):
    return state.estimate_values.index(max(state.estimate_values))

def e_greedy(epsilon, state: narmedbandit.State):
    x = random.random()

    if (x < epsilon):
        return random.randrange(0, state.action_space_size)
    
    return state.estimate_values.index(max(state.estimate_values))