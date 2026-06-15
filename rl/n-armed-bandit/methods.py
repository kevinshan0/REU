import narmedbandit

def sample_average(state: narmedbandit.State, action: int, reward: float):
    return state.estimate_values[action] + (1 / state.action_count[action]) * (reward - state.estimate_values[action])

def weighted_average(alpha: float, state: narmedbandit.State, action: int, reward: float):
    return state.estimate_values[action] + alpha * (reward - state.estimate_values[action])