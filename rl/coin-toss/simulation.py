from dataclasses import dataclass
import random

from policy_iteration import iter
from policy_evaluation import eval
import matplotlib.pyplot as plt

# config
WIN_CONDITION = 100
P_HEADS = 0.4
GAMMA = 1.0
DELTA = 1e-9

@dataclass
class Game:
    balance: int
    p_heads: float
    policy: list[int]

    def run(self):
        while self.balance != 0 and self.balance != WIN_CONDITION:
            bet = self.policy[self.balance - 1]
            self.balance += bet if random.random() < self.p_heads else -bet

        return self.balance == WIN_CONDITION


def plot_policy_and_values(policy, state_values, win_condition=100):
    states = list(range(1, win_condition))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(states, policy, marker='o', linestyle='-', color='tab:blue', label='Policy Bet')
    ax1.set_xlabel('Balance')
    ax1.set_ylabel('Bet Amount', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(states, state_values, marker='x', linestyle='--', color='tab:red', label='State Value')
    ax2.set_ylabel('State Value', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    fig.suptitle('Policy Bet and State Value vs Balance', y=1.02)
    fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.95))
    plt.show()


policy, state_values = iter(p=P_HEADS, gamma=GAMMA, max_change=DELTA, win_condition=WIN_CONDITION)
plot_policy_and_values(policy, state_values, WIN_CONDITION)
