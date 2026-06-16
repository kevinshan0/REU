# a gambler can bet that a coin toss lands on heads with probability p that the coin lands on heads and (1 - p) that lands on tails iwht p > 0.5
# the gambler wants to reach 100, the terminal state with reward 1, starting from 10
# the states consist of {1, 2, 3, ..., 99} representing the gambler's capital with corresponding actions at state s {0, 1, 2, ..., min(s, 100 - s)}.
# the min(s, 100 - s) is the natural decision to not risk more than required to achieve the goal of 100 from state s.

import math

def eval(state_values=None, policy=None, p=0.6, gamma=0.9, max_change=0.01, win_condition=100):
    if policy is None:
        policy = [0] * (win_condition - 1)
    if state_values is None:
        state_values = [0.0] * (win_condition - 1)

    while True:
        delta = 0
        for state_index, _ in enumerate(state_values):
            prev_state_value = state_values[state_index]
            state = state_index + 1
            action = policy[state_index]
            s1 = state + action
            s2 = state - action
            s1_reward = 1 if s1 == win_condition else 0

            if s1 == win_condition:
                v_win = s1_reward  # terminal
            else:
                v_win = s1_reward + gamma * state_values[s1 - 1]

            if s2 == 0:
                v_lose = 0  # terminal
            else:
                v_lose = gamma * state_values[s2 - 1]
                
            state_values[state_index] = p * v_win + (1 - p) * v_lose
            delta = max(delta, abs(state_values[state_index] - prev_state_value))

        if delta < max_change:
            break

    return state_values