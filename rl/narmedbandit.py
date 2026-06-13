# 10-arm bandit problem
from dataclasses import dataclass

import numpy as np
from collections.abc import Callable

# state data structure
@dataclass
class State:
    def __init__(self, action_space_size: int, steps: int, initial_estimates: float):
        self.action_space_size = action_space_size
        self.steps = steps
        self.real_values = []
        self.action_count = [0] * action_space_size
        self.estimate_values = [initial_estimates] * action_space_size

        # sample from the normal distribution for true values
        for _ in range(0, action_space_size):
            self.real_values.append(np.random.normal())

    def update_state(self, action: int, update_rule: Callable) -> float:
        self.action_count[action] += 1
        reward = self.real_values[action] + np.random.normal()
        self.estimate_values[action] = update_rule(self, action, reward)
        return reward

def run_trials(
        trials: int,
        action_space_size: int,
        steps: int,
        update_rule: Callable,
        initial_estimates: int,
        algorithm: Callable,
        observer: Callable,
    ):
    for _ in range(trials):
        # initialize state and run algorithm
        state = State(action_space_size, steps, initial_estimates)

        for step in range(steps):
            action = algorithm(state)
            reward = state.update_state(action, update_rule)
            observer(step, action, reward)