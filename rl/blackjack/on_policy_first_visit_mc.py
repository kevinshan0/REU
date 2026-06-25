import environment
import collections

action_values = {}
returns = collections.defaultdict(list)
policy = {}
epsilon = 0.1

TRIALS = 100000

def init():
    # arbitrary action-value function
    # arbitrary e-soft policy, start at 50/50 hit/stand rate at any given state
    for agent_sum in range(12, 22):
        for dealer_sum in range(1, 11):
            action_values[(agent_sum, dealer_sum), True] = 0
            action_values[(agent_sum, dealer_sum), False] = 0
            policy[(agent_sum, dealer_sum), True] = 0.5
            policy[(agent_sum, dealer_sum), False] = 0.5

    rewards = []

    for _ in range(0, TRIALS):
        episode = environment.Episode()
        rewards.append(episode.run_episode(policy, action_values, returns, epsilon))
        print(sum(rewards) / len(rewards))

init()