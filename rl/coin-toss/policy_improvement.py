import numpy as np

def impr(state_values=None, policy=None, p=0.6, gamma=0.9, win_condition=100):
    if policy is None:
        policy = [0] * (win_condition - 1)
    if state_values is None:
        state_values = [0.0] * (win_condition - 1)

    policy_stable = True
    for state_index, _ in enumerate(policy):
        prev_action = policy[state_index]
        state = state_index + 1
        actions = list(range(0, min(state, win_condition - state) + 1))
        action_values = []
        for action in actions:
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

            action_values.append(p * v_win + (1 - p) * v_lose)

        qs = np.array(action_values)
        policy[state_index] = int(np.argmax(np.round(qs[1:], 5)) + 1) 
        if prev_action != policy[state_index]: 
            policy_stable = False
    
    return policy_stable