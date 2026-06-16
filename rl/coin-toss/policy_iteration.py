from policy_evaluation import eval
from policy_improvement import impr

def iter(p=0.6, gamma=0.9, max_change=0.01, win_condition=100):
    state_values = [0.0] * (win_condition - 1)
    policy = [0] * (win_condition - 1)
    policy_stable = False

    while not policy_stable:
        state_values = eval(state_values, policy, p=p, gamma=gamma, max_change=max_change, win_condition=win_condition)
        policy_stable = impr(state_values, policy, p=p, gamma=gamma, win_condition=win_condition)

    return policy, state_values